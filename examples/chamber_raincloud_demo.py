"""A raincloud in a room, from the laws alone.

A 2 m column of warm, saturated air (8 cells of 0.25 m) under a cold plate.
The plate cools the top cell through the surface law's convection; the air
step conducts it down; each cell that crosses its dew point condenses
(species law); cloud water above the autoconversion threshold becomes rain,
accretes, falls cell to cell, and lands on the floor surface, whose runoff
fills the pool.  Nothing in here says "fog", "cloud" or "rain".

    python examples/chamber_raincloud_demo.py [seconds]
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


laws = load("symbolic_chamber_solvers")
load("chamber_dt_join")
sim_mod = load("chamber_sim")

from src.common.dt_system.dt_controller import STController, Targets, run_superstep  # noqa: E402


def esat(T):
    return 611.657 * math.exp((2.5e6 / 461.5) * (1 / 273.16 - 1 / T))


WATER = dict(P_tp=611.657, T_tp=273.16, L_v=2.5e6, L_f=3.34e5, R_v=461.5, R_a=287.0, R_u=8.314,
             M_w=0.018015, rho_w=1000.0, rho_ice=917.0, sigma_w=0.072, D_v=2.5e-5, k_a=0.026,
             g=9.81, mu_air=1.8e-5, rho_air=1.2, Cd=0.47, cv_a=718.0, cv_v=1410.0, c_w=4186.0, c_i=2100.0)
NACL = dict(M_s=0.05844, rho_s=2165.0, i_vh=2.0, kappa=1.28)


def build(nz=8, dx=0.25, T0=295.0, T_plate=270.0):
    n = nz
    V = dx ** 3
    m_a = 1.2 * V
    m_v = esat(T0) * V / (461.5 * T0)          # saturated everywhere
    col = sim_mod.column
    air_state = {"m_a": col([m_a] * n), "T": col([T0] * n)}
    air_params = dict(R_a=287.0, cv_a=718.0, k_a=0.026, k_flow=1e-4, dP_blend=1.0)
    water_state = {"m_v": col([m_v] * n), "m_l": col([0.0] * n), "m_i": col([0.0] * n), "m_r": col([0.0] * n)}
    water_params = dict(P_tp=611.657, T_tp=273.16, L_v=2.5e6, L_f=3.34e5, R_v=461.5, D_v=2.5e-5,
                        cv_v=1410.0, c_w=4186.0, c_i=2100.0,
                        q_c0=5e-4, k_auto=1e-2, k_acc=2.2, k_revap=1e-3, w_l=0.02, w_i=0.3, w_r=4.0,
                        w_l_up=0.02, w_i_up=0.3, w_r_up=4.0, F_blend=1e-9, tau_cond=0.5, tau_freeze=5.0)
    surface_params = {**WATER, **NACL, "A_s": dx * dx, "U_plate": 200.0, "h_conv": 15.0, "C_s": 5000.0,
                      "alpha_evap": 0.04, "s_drop": 0.0, "phi_sed": 0.0, "k_wash": 0.2, "tau_dens": 3600.0,
                      "tau_cryst": 10.0, "k_frost": 0.1, "rho_sed": 1500.0, "b_sat0": 6.1, "db_sat_dT": 0.002,
                      "T_ref": 298.15, "sigma_l": 0.072, "theta_c": 1.2, "mu_l": 1e-3, "w_edge": dx}
    surface_state = lambda T_s: {"m_film": col([1e-9]), "n_s_film": col([0.0]), "m_frost": col([0.0]),  # noqa: E731
                                 "rho_frost": col([100.0]), "m_crust": col([0.0]), "m_sed": col([0.0]),
                                 "T_s": col([T_s])}
    ceiling = sim_mod.SurfaceSpec(voxel=n - 1, state=surface_state(T_plate),
                                  params={**surface_params, "T_plate": T_plate, "tilt": 0.0, "dcos_hyst": 0.1,
                                          "phi_drop": 0.0})
    floor = sim_mod.SurfaceSpec(voxel=0, state=surface_state(T0), pool=0, catches_rain=True,
                                params={**surface_params, "T_plate": T0, "tilt": 0.3, "dcos_hyst": 0.05})
    pool = sim_mod.PoolSpec(voxel=0, state={"m_p": col([1e-6]), "n_s_pool": col([0.0]), "T_l": col([T0])},
                            params={**WATER, **NACL, "A_floor": dx * dx, "T_in": T0, "h_sill": 0.05, "w_sill": dx,
                                    "C_w": 0.4, "T_floor": T0, "U_floor": 50.0, "h_conv": 5.0, "alpha_evap": 0.04,
                                    "sigma_l": 0.072, "theta_c": 1.2, "mu_l": 1e-3})
    return sim_mod.ChamberSim(laws, shape=(1, 1, nz), dx=dx, air_state=air_state, air_params=air_params,
                              species={"water": (water_state, water_params)},
                              surfaces=[ceiling, floor], pools=[pool])


def main(seconds=60.0):
    sim = build()
    targets = Targets(cfl=0.5, div_max=1e9, mass_max=1e-3, energy_exchange_fraction=0.2)
    ctrl = STController()
    t = 0.0
    dt = 0.01
    fmt = lambda col: " ".join(f"{float(v):9.2e}" for v in col.tolist())  # noqa: E731
    print("      t     T_top   T_bot |  LWC per cell (top..bottom)                | rain_out bottom | pool kg")
    while t < seconds:
        total, dt, metrics = run_superstep(sim.state, 1.0, dt, sim.dx, targets, ctrl, sim.advance)
        t += float(total)
        water = sim.species["water"].state
        T = sim.air.state.columns["T"].tolist()
        lwc = list(reversed(water.outputs["LWC"].tolist()))
        rain = float(water.outputs["rain_out"][[0]].item())
        pool = float(sim.pools[0][1].state.columns["m_p"].item())
        print(f"{t:7.2f} {T[-1]:8.2f} {T[0]:7.2f} | {fmt(sim_mod.column(lwc))} | {rain:9.2e} | {pool:9.3e}"
              f"   dt={float(dt):.4f} dt_limit={metrics.dt_limit:.4f}")
    return sim


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
