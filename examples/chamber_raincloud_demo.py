"""A raincloud in a room: PARAMETERS AND RUNTIME EXPERIENCE ONLY.

===========================================================================
CAUTION -- READ BEFORE ADDING ANYTHING TO THIS FILE
===========================================================================

This file is NOT allowed to contain simulation code.  No laws, no stepping,
no state construction, no controller loop, no interpreter of any kind.

What happened here (2026-09-17): this file used to build a ``ChamberSim``
over the SymPy laws and step it through ``run_superstep`` itself.  The laws
were lowered to the repository's dual IR and then executed by
``_abstract_tensor_stage_callable`` -- the Python-side reference executor
whose only purpose is parity against the native lane.  It has NO runtime
purpose.  Because this file wired that up under the name "the demo", a
whole session was spent stepping the raincloud through the interpreter
(~0.1 s per substep for 8 cells) while calling it "compiled", and none of
it was on the path that is actually being compiled: ``llvm_dt_system.py``,
the dt system with LLVM pieces as its steps.

So the rule, absolute:

  * The simulation is produced by the sanctioned lane -- native pieces
    (``llvm_dt_system.dt_system`` over ``LLVMPiece`` files) or, at minimum,
    lambdified SymPy.  Never the AbstractTensor interpreter.  Ever.
  * This file DECLARES the scenario (the parameters below) and OWNS the
    runtime experience (what a person sees: the table, the viewer hookup).
    It does not compute physics and it does not advance time.
  * If you find yourself importing a law module, building state columns,
    or calling ``run_superstep`` from here, stop -- that belongs in the
    lane, not in the scenario.

The scenario, in words: a 2 m column of warm, saturated air (8 cells of
0.25 m) under a cold plate.  The plate cools the top cell through the
surface law's convection; the air step conducts it down; each cell that
crosses its dew point condenses (species law); cloud water above the
autoconversion threshold becomes rain, accretes, falls cell to cell, and
lands on the floor surface, whose runoff fills the pool.  Nothing in the
laws says "fog", "cloud" or "rain".

    python examples/chamber_raincloud_demo.py [seconds]
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# PARAMETERS.  Data only.  Every number a law needs to run this scenario.
# ---------------------------------------------------------------------------

#: Chamber geometry.  ``shape`` is (nx, ny, nz) with z vertical; the flat
#: index the laws use is ``x + nx*(y + ny*z)``.
GEOMETRY = dict(shape=(1, 1, 8), dx=0.25)

#: Initial conditions.  The air starts SATURATED everywhere (relative
#: humidity 1.0) at T0; the plate is held at T_plate.
INITIAL = dict(T0=295.0, T_plate=270.0, relative_humidity=1.0, rho_air=1.2)

#: Water, the condensable species: triple point, latent heats, gas constants,
#: transport and thermal properties.
WATER = dict(P_tp=611.657, T_tp=273.16, L_v=2.5e6, L_f=3.34e5, R_v=461.5, R_a=287.0, R_u=8.314,
             M_w=0.018015, rho_w=1000.0, rho_ice=917.0, sigma_w=0.072, D_v=2.5e-5, k_a=0.026,
             g=9.81, mu_air=1.8e-5, rho_air=1.2, Cd=0.47, cv_a=718.0, cv_v=1410.0, c_w=4186.0, c_i=2100.0)

#: Sodium chloride, the solute (sea salt / road salt): molar mass, density,
#: van 't Hoff dissociation count, hygroscopicity.
NACL = dict(M_s=0.05844, rho_s=2165.0, i_vh=2.0, kappa=1.28)

#: voxel_air_step
AIR_PARAMS = dict(R_a=287.0, cv_a=718.0, k_a=0.026, k_flow=1e-4, dP_blend=1.0)

#: voxel_species_step (water).
WATER_PARAMS = dict(P_tp=611.657, T_tp=273.16, L_v=2.5e6, L_f=3.34e5, R_v=461.5, D_v=2.5e-5,
                    cv_v=1410.0, c_w=4186.0, c_i=2100.0,
                    q_c0=5e-4, k_auto=1e-2, k_acc=2.2, k_revap=1e-3, w_l=0.02, w_i=0.3, w_r=4.0,
                    w_l_up=0.02, w_i_up=0.3, w_r_up=4.0, F_blend=1e-9, tau_cond=0.5, tau_freeze=5.0)

#: surface_step, shared by the ceiling plate and the floor.  ``A_s`` and
#: ``w_edge`` are dx^2 and dx for this geometry.
SURFACE_PARAMS = {**WATER, **NACL, "A_s": GEOMETRY["dx"] ** 2, "U_plate": 200.0, "h_conv": 15.0, "C_s": 5000.0,
                  "alpha_evap": 0.04, "s_drop": 0.0, "phi_sed": 0.0, "k_wash": 0.2, "tau_dens": 3600.0,
                  "tau_cryst": 10.0, "k_frost": 0.1, "rho_sed": 1500.0, "b_sat0": 6.1, "db_sat_dT": 0.002,
                  "T_ref": 298.15, "sigma_l": 0.072, "theta_c": 1.2, "mu_l": 1e-3, "w_edge": GEOMETRY["dx"]}

#: The two surfaces: which voxel each sits on, its own plate temperature,
#: tilt and contact-angle hysteresis, and whether it catches what falls.
SURFACES = (
    dict(name="ceiling", voxel="top", T_plate=INITIAL["T_plate"], tilt=0.0, dcos_hyst=0.1, phi_drop=0.0,
         catches_rain=False, pool=None),
    dict(name="floor", voxel="bottom", T_plate=INITIAL["T0"], tilt=0.3, dcos_hyst=0.05,
         catches_rain=True, pool=0),
)

#: Seed state of every surface: a nominal 1e-9 kg film so the film laws
#: have a denominator, no frost, no crust, no sediment, at its own T_plate.
SURFACE_SEED = dict(m_film=1e-9, n_s_film=0.0, m_frost=0.0, rho_frost=100.0, m_crust=0.0, m_sed=0.0)

#: pool_step: the floor's pool.
POOL_PARAMS = {**WATER, **NACL, "A_floor": GEOMETRY["dx"] ** 2, "T_in": INITIAL["T0"], "h_sill": 0.05,
               "w_sill": GEOMETRY["dx"], "C_w": 0.4, "T_floor": INITIAL["T0"], "U_floor": 50.0, "h_conv": 5.0,
               "alpha_evap": 0.04, "sigma_l": 0.072, "theta_c": 1.2, "mu_l": 1e-3}
POOL_SEED = dict(m_p=1e-6, n_s_pool=0.0)

#: The dt system's targets and opener for this scenario.
DT = dict(cfl=0.5, div_max=1e9, mass_max=1e-3, energy_exchange_fraction=0.2, round_s=1.0, dt_initial=0.01)


# ---------------------------------------------------------------------------
# RUNTIME EXPERIENCE.  What a person sees.  No physics below this line.
# ---------------------------------------------------------------------------

HEADER = "      t     T_top   T_bot |  LWC per cell (top..bottom)                | rain_out bottom | pool kg"


def format_row(t, T_top, T_bot, lwc_top_to_bottom, rain_out_bottom, pool_kg, dt, dt_limit) -> str:
    """One line of the table, from numbers the runtime hands in."""
    lwc = " ".join(f"{float(v):9.2e}" for v in lwc_top_to_bottom)
    dtl = float("nan") if dt_limit is None else float(dt_limit)
    return (f"{t:7.2f} {T_top:8.2f} {T_bot:7.2f} | {lwc} | {rain_out_bottom:9.2e} | {pool_kg:9.3e}"
            f"   dt={float(dt):.4f} dt_limit={dtl:.4f}")


def main(seconds: float = 60.0) -> None:
    """Run the scenario through the sanctioned lane and show the table.

    There is no simulation here to run.  Until the chamber's laws are
    emitted as native pieces and stepped by ``llvm_dt_system.dt_system``,
    this entry point refuses rather than falling back to the interpreter.
    """
    sys.exit(
        "chamber_raincloud_demo: no native runtime is wired for this scenario yet.\n"
        "  The parameters are declared above; the stepping belongs to examples/llvm_dt_system.py\n"
        "  (LLVM pieces as the dt system's steps).  This file will not run the AbstractTensor\n"
        "  interpreter -- see the CAUTION at the top of the file."
    )


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
