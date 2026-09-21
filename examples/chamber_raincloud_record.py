"""Record the raincloud demo's published state, superstep by superstep.

The demo (chamber_raincloud_demo.py) is the headless run: the SAME build,
the SAME controller, the SAME loop, unchanged.  This wraps it and writes
what each law PUBLISHES -- every ``LAW_PUBLICATIONS`` output with its
declared semantic and unit -- to one ``.npz`` the viewer plays back.

Why record rather than render live: the chamber advances ~205 substeps
per simulated second (dt pinned at the laws' own stability limit), each
evaluating seven compiled laws, which is roughly twenty times slower than
real time on this machine.  A window stepping that directly would show a
few hundredths of a second of weather per wall second.  Recording once
and playing back at any rate -- scrubbable, loopable -- also frees the
renderer from the sim's construction cost, and means a better run later
(the debt ledger, the aerosol law switched on) is a new file, not a new
viewer.

Channels are keyed by the law output NAME, and the file carries the
declared semantic/unit for each so the viewer can look up "cloud water
content" rather than guess from a variable name.

    python examples/chamber_raincloud_record.py [seconds] [out.npz]
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def _load_demo():
    spec = importlib.util.spec_from_file_location("chamber_raincloud_demo", HERE / "chamber_raincloud_demo.py")
    demo = importlib.util.module_from_spec(spec)
    sys.modules["chamber_raincloud_demo"] = demo
    spec.loader.exec_module(demo)
    return demo


def _col(value) -> np.ndarray:
    """An AbstractTensor column (or float) as a flat float64 array."""
    if hasattr(value, "tolist"):
        return np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
    return np.asarray([float(value)], dtype=np.float64)


def build_native(demo, laws, shape: tuple[int, int, int], *,
                 plate_temperature: float | None = None,
                 top_surface_areas: dict[int, float] | None = None):
    """Build the declared chamber over the existing LLVM pieces."""
    from chamber_dt_join import CompiledLaw
    from chamber_sim import ChamberSim, PoolSpec, SurfaceSpec, column
    from src.compiler.native_law_kernels import LLVMPiece

    nx, ny, nz = shape
    n = nx * ny * nz
    dx = float(demo.GEOMETRY["dx"])
    T0 = float(demo.INITIAL["T0"])
    T_plate = float(demo.INITIAL["T_plate"] if plate_temperature is None else plate_temperature)
    V = dx ** 3
    m_a = float(demo.INITIAL["rho_air"]) * V
    e_sat = float(demo.WATER["P_tp"]) * math.exp(
        (float(demo.WATER["L_v"]) / float(demo.WATER["R_v"]))
        * (1.0 / float(demo.WATER["T_tp"]) - 1.0 / T0)
    )
    m_v = e_sat * V / (float(demo.WATER["R_v"]) * T0)
    air_state = {"m_a": column([m_a] * n), "T": column([T0] * n)}
    water_state = {
        "m_v": column([m_v] * n), "m_l": column([0.0] * n),
        "m_i": column([0.0] * n), "m_r": column([0.0] * n),
    }
    surface_state = lambda T_s: {  # noqa: E731
        **{name: column([value]) for name, value in demo.SURFACE_SEED.items()},
        "T_s": column([T_s]),
    }
    surfaces = []
    top_cells = {
        x + nx * (y + ny * (nz - 1)): dx * dx
        for y in range(ny) for x in range(nx)
    } if top_surface_areas is None else dict(top_surface_areas)
    for top, area in top_cells.items():
        surfaces.append(SurfaceSpec(
            voxel=top, state=surface_state(T_plate),
            params={**demo.SURFACE_PARAMS, "A_s": float(area),
                    "T_plate": T_plate,
                    "tilt": 0.0, "dcos_hyst": 0.1, "phi_drop": 0.0},
        ))
    for y in range(ny):
        for x in range(nx):
            bottom = x + nx * (y + ny * 0)
            surfaces.append(SurfaceSpec(
                voxel=bottom, state=surface_state(T0), pool=0,
                catches_rain=True,
                params={**demo.SURFACE_PARAMS, "T_plate": T0,
                        "tilt": 0.3, "dcos_hyst": 0.05},
            ))
    pool = PoolSpec(
        voxel=0,
        state={"m_p": column([demo.POOL_SEED["m_p"]]),
               "n_s_pool": column([demo.POOL_SEED["n_s_pool"]]),
               "T_l": column([T0])},
        params={**demo.POOL_PARAMS, "A_floor": dx * dx * nx * ny},
    )
    pieces = HERE.parent / "artifacts" / "llvm_pieces"
    load = lambda law, batch: CompiledLaw.from_piece(LLVMPiece.load(  # noqa: E731
        pieces / law / f"b{batch}" / f"{law}.piece"
    ))
    compiled = {
        "voxel_air_step": load("voxel_air_step", n),
        "voxel_species_step": load("voxel_species_step", n),
        "surface_step": load("surface_step", 1),
        "pool_step": load("pool_step", 1),
    }
    return ChamberSim(
        laws, shape=(nx, ny, nz), dx=dx,
        air_state=air_state, air_params=dict(demo.AIR_PARAMS),
        species={"water": (water_state, dict(demo.WATER_PARAMS))},
        surfaces=surfaces, pools=[pool], compiled_laws=compiled,
    )


def record(seconds: float = 60.0, out: str | Path = HERE / "chamber_raincloud_run.npz",
           *, log_every: float = 1.0, shape: tuple[int, int, int] | None = None) -> Path:
    demo = _load_demo()
    from chamber_dt_join import load_law_module_cached
    from src.common.dt_system.dt_controller import STController, Targets, run_superstep

    laws = load_law_module_cached(HERE / "symbolic_chamber_solvers.py")
    sim = build_native(demo, laws, shape or tuple(demo.GEOMETRY["shape"]))
    nx, ny, nz = sim.shape

    # semantic + unit per published output, from the laws' own declaration
    publications = {}
    for law_name, rows in getattr(laws, "LAW_PUBLICATIONS", {}).items():
        for row in rows:
            publications[row.output] = {"law": law_name, "semantic": row.semantic, "unit": row.unit}

    targets = Targets(cfl=0.5, div_max=1e9, mass_max=1e-3, energy_exchange_fraction=0.2)
    ctrl = STController()
    t, dt = 0.0, 0.01

    water = sim.species[sim.water]
    frames: dict[str, list] = {}

    def put(name, value):
        frames.setdefault(name, []).append(np.asarray(value, dtype=np.float64))

    t_wall = time.time()
    next_log = 0.0
    while t < seconds:
        log: list = []
        total, dt, metrics = run_superstep(sim.state, 1.0, dt, sim.dx, targets, ctrl, sim.advance,
                                           attempt_log=log)
        t += float(total)
        rejected = sum(1 for a in log if not a["accepted"])

        put("t", t)
        put("dt", float(dt))
        put("dt_limit", float("nan") if metrics.dt_limit is None else float(metrics.dt_limit))
        put("substeps", len(log))
        put("rejected", rejected)
        put("mass_err", float(metrics.mass_err))

        # air: the one shared temperature and pressure per voxel
        put("T", _col(sim.air.state.columns["T"]))
        for name in ("P", "rho_a"):
            if name in sim.air.state.outputs:
                put(name, _col(sim.air.state.outputs[name]))
        # water species: every published voxel field, plus the raw masses
        for name in ("S", "LWC", "IWC", "RWC", "rain_out", "drizzle_out", "snow_out", "cond_rate", "auto_rate"):
            if name in water.state.outputs:
                put(name, _col(water.state.outputs[name]))
        for name in ("m_v", "m_l", "m_i", "m_r"):
            put(name, _col(water.state.columns[name]))
        # surfaces and pools: one value each, kept per index
        for i, (spec, engine) in enumerate(sim.surfaces):
            for name in ("h_film", "h_frost", "crust_thickness", "sediment_thickness"):
                if name in engine.state.outputs:
                    put(f"surface{i}.{name}", _col(engine.state.outputs[name]))
            put(f"surface{i}.T_s", _col(engine.state.columns["T_s"]))
        for i, (spec, engine) in enumerate(sim.pools):
            for name in ("h_pool", "A_wet", "overflow_rate"):
                if name in engine.state.outputs:
                    put(f"pool{i}.{name}", _col(engine.state.outputs[name]))
            put(f"pool{i}.m_p", _col(engine.state.columns["m_p"]))

        if t >= next_log:
            lwc = _col(water.state.outputs["LWC"]) if "LWC" in water.state.outputs else np.zeros(sim.n)
            print(f"  t={t:7.2f}  substeps={len(log):4d} rejected={rejected:4d}  dt={float(dt):.4f}  "
                  f"LWC top={lwc[-1]:.2e}  wall={time.time() - t_wall:6.1f} s", flush=True)
            next_log += log_every

    arrays = {name: np.stack(vals) for name, vals in frames.items()}
    meta = {
        "shape": [int(nx), int(ny), int(nz)], "dx": float(sim.dx), "vertical_axis": "z",
        "flat_index": "x + nx*(y + ny*z)",
        "surfaces": [{"voxel": int(s.voxel), "catches_rain": bool(s.catches_rain), "pool": s.pool,
                      "T_plate": float(s.params.get("T_plate", float("nan")))} for s, _e in sim.surfaces],
        "pools": [{"voxel": int(p.voxel)} for p, _e in sim.pools],
        "publications": publications,
        "seconds": float(t), "n_frames": int(len(frames["t"])),
    }
    out = Path(out)
    np.savez_compressed(out, __meta__=json.dumps(meta), **arrays)
    print(f"recorded {meta['n_frames']} frames, {t:.1f} s of weather -> {out}  ({time.time() - t_wall:.0f} s wall)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("seconds", nargs="?", type=float, default=60.0)
    ap.add_argument("out", nargs="?", default=str(HERE / "chamber_raincloud_run.npz"))
    ap.add_argument("--shape", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=None,
                    help="widen the chamber (default: the demo's own (1,1,8) build, unchanged)")
    a = ap.parse_args()
    record(a.seconds, a.out, shape=tuple(a.shape) if a.shape else None)
