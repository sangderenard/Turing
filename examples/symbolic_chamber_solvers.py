"""Thermodynamic-chamber solvers as simultaneous SymPy state-transition laws.

Companion to symbolic_atmosphere_model.py.  That file states the point
relations (Clausius-Clapeyron, Koehler, Hertz-Knudsen, Beer-Lambert); this
file turns them into the state machines a chamber game steps every tick,
in the form compile_sympy_equations() accepts: one call is ONE simultaneous
transition, every output is a fresh name (``*_next`` or a diagnostic), and
no output appears on any right-hand side.  Recurrence -- feeding ``x_next``
back in as ``x`` -- is the caller's loop, never the law's.

LAWS
  voxel_air_step      Eulerian cell, dry-air carrier: ideal-gas mixture
                      pressure, pressure-driven exchange with six face
                      neighbours (donor-cell upwind without a branch), heat
                      conduction, one shared temperature fed by every
                      species' latent heat.
  voxel_species_step  One condensable species in that cell, run once per
                      species: vapour advected on the air step's face flows
                      and diffusing, relaxation to ITS OWN saturation curve
                      (fog / cloud appear at its dew point), freezing /
                      melting, Kessler warm rain (autoconversion, accretion,
                      rain evaporation), settling of cloud, ice and rain.
  aerosol_step        Smoke / dust / salt haze in that cell as two moments,
                      number and dry mass per m^3 (analytic, never discrete):
                      emission, Brownian coagulation, advection on the air
                      step's face flows, size-dependent settling, Brownian
                      wall deposition, rain washout, and kappa-Koehler CCN
                      activation, which publishes the droplet number and the
                      condensation time the species step takes as
                      ``tau_cond`` -- so the aerosol seeds the cloud that
                      rains it out.  Publishes extinction, Mie / Rayleigh
                      weight and transmittance for the renderer.
  droplet_step        Lagrangian particle: kappa-Koehler equilibrium over a
                      dissolved inorganic salt, Mason diffusional growth and
                      evaporation, dry-out to the salt crystal (a radius
                      floor, no branch), latent heating of the drop, Stokes /
                      Newton terminal velocity, velocity relaxation and fall.
  salt_solution_step  Bulk solution of one inorganic salt: dissolved versus
                      crystalline moles, solubility versus temperature,
                      supersaturation-driven crystal growth, nucleation and
                      dissolution, Raoult water activity, freezing-point
                      depression, boiling-point elevation, deliquescence /
                      efflorescence hysteresis, water uptake toward the
                      hygroscopic equilibrium.
  surface_step        A bounding surface: dew film, frost layer (with
                      densification and its own thermal resistance), salt
                      crust precipitating out of the drying film and
                      redissolving when re-wetted, sediment deposit, and the
                      surface heat balance against the gas and the plate
                      behind it.  The film is held by surface tension at the
                      sessile thickness 2 l_c sin(theta/2) and the excess
                      drains as a Nusselt gravity film, so what leaves is a
                      mass rate set by sigma, theta, mu, tilt and edge width.
  pool_step           A liquid body: mass, solute and temperature, with its
                      geometry derived, not stored -- it spreads at the
                      sessile thickness until the floor is covered, then
                      deepens; it evaporates from its wetted area, takes
                      inflow from rain and runoff, overflows a sill as a
                      weir, and exchanges heat with the air and the floor.

FLUID PAIRING
  Liquid is a mass, a solute count and a temperature everywhere in these
  laws; area, thickness and depth are derived from sigma / theta / rho / g
  on demand.  A surface-tension model supplies ``sigma_l`` and ``theta_c``
  (per pair of liquid and material) and reads back ``h_cap`` / ``A_wet`` /
  ``h_pool``; a fluid-volume model takes ``runoff`` and ``overflow`` as mass
  rates and supplies ``phi_in``.  The condensable's own ``M_w`` / ``rho_w``
  are inputs, so a heavy liquid is the same law with its own numbers.

Every law publishes ``dt_limit``: the largest explicit step for which its
own update is stable, so the caller's clock can honour the tightest one.
It is a closed-form function of the law's parameters: diffusion and flow
give CFL terms, every explicit rate term gives ``1 / |d rate / d state|``
from the symbolic Jacobian (``euler_bound``), and terms integrated exactly
(``relax``, velocity and drop-temperature relaxation) impose none.

Every law also publishes the dt system's own diagnostics, named as
``dt_scaler.Metrics`` names them so the join is a lookup, not a translation:
``max_vel`` (the fastest transport speed in the cell), ``max_flux`` (mass
moved per second), ``div_inf`` (relative transport divergence), ``mass_err``
(mass the positivity floors refused -- exactly zero when no clamp bit), and
the ``energy_j`` / ``power_w`` channels the controller uses to pin one step
to a fraction of the stored energy.
Coupling between laws is by value, in the caller: a voxel's ``rain_out``
becomes a floor surface's ``phi_drop``; a surface's ``runoff`` becomes the
``phi_drop`` of the surface below it; a droplet's ``v_t`` is a voxel's
``w_l``; a voxel's ``e`` and ``T`` are every particle's ambient; the air
step's ``F_*`` are every species step's face flows, and the species steps'
``Q_lat`` / ``C_species`` / ``e`` summed are the air step's ``Q_ext`` /
``C_cond`` / ``P_vap``.

CONVENTIONS (same as engine_toy/symbolic_atmosphere.py)
  sympy.Max / sympy.Min for clamps, never (x + Abs(x))/2.
  smooth_abs / smooth_step are branch-free (sqrt(x*x + eps)); no sign().
  Species and material constants are INPUTS, never literals, so the same
  law runs any salt or any condensable straight from a table.
  Every column is one voxel / particle / surface; the law is batched.
"""

from __future__ import annotations

import sympy as sp

from src.compiler.symbolic_equation_compiler import SymbolicPublication

pi = sp.pi
EPS = sp.Float(1e-12)      # floor for O(1) quantities (ratios, seconds, kg of bulk)
TINY = sp.Float(1e-40)     # floor for SI particle quantities (m^3, kg, J/K of one drop)
HALF = sp.Rational(1, 2)


def interface_flux(P_sat, e, T_s, alpha, h_conv, M, R_v_, rho_gas, cp_gas):
    """Net evaporation flux kg/(m^2 s), positive leaving the surface.

    Two resistances in series: the kinetic (Hertz-Knudsen) limit at the
    interface and boundary-layer diffusion by the Chilton-Colburn analogy
    (mass-transfer coefficient h / (rho cp), Lewis number 1).  Kinetic alone
    is the vacuum limit and overstates a room flux by ~10^4.
    """
    R_kin = sp.sqrt(2 * pi * M * R_u * T_s) / alpha
    R_diff = R_v_ * T_s * rho_gas * cp_gas / h_conv
    return (P_sat - e) / (R_kin + R_diff)


def smooth_abs(x):
    return sp.sqrt(x * x + TINY)


def smooth_step(x, width):
    return HALF * (1 + x / sp.sqrt(x * x + width * width))


def clamp01(x):
    return sp.Min(sp.Max(x, 0), 1)


def hard_step(x, width):
    """Cubic smoothstep: exactly 0 below -width and exactly 1 above +width.

    For hysteresis states, where a leak of even 1e-3 per tick would drift
    the state across the band on its own.
    """
    u = clamp01((x + width) / (2 * width))
    return u * u * (3 - 2 * u)


def euler_bound(rate, state):
    """Largest dt with no overshoot for explicit Euler on d(state)/dt = rate.

    ``1 / |d rate / d state|``, derived symbolically from the law's own
    parameters.  Kinks from Max/Min differentiate to Heaviside, which has no
    native lowering; it is spelled as a narrow smooth step instead.
    """
    slope = sp.diff(rate, state).replace(
        sp.Heaviside, lambda *args: smooth_step(args[0], sp.Float(1e-9)))
    return 1 / (smooth_abs(slope) + EPS)


def relax(dt, tau):
    """Fraction of a gap closed in ``dt`` by first-order relaxation.

    Spelled 2 e^(-x/2) sinh(x/2), which is 1 - e^(-x) without the
    cancellation that costs 1 - e^(-x) two digits when x is small.
    """
    x = dt / (tau + EPS)
    return 2 * sp.exp(-x / 2) * sp.sinh(x / 2)


def cbrt(x):
    return sp.Pow(x, sp.Rational(1, 3))


def saturation_pressure(T, P_tp, T_tp, L, R_v):
    """Clausius-Clapeyron integrated at constant latent heat from the triple point.

    The exponent is (T - T_tp) / (T T_tp), not 1/T_tp - 1/T: the difference
    of two reciprocals cancels to a few digits and L / R_v then amplifies
    that loss into e_s, S, and everything condensation drives.
    """
    return P_tp * sp.exp((L / R_v) * (T - T_tp) / (T * T_tp))


def named(name, expr):
    return sp.Eq(sp.Symbol(name), expr, evaluate=False)


# Shared inputs (same meaning in every law they appear in).
dt = sp.Symbol("dt")
T, P_tp, T_tp, L_v, L_f, R_v, R_a, R_u = sp.symbols("T P_tp T_tp L_v L_f R_v R_a R_u")
M_w, rho_w, rho_ice, sigma_w, D_v, k_a = sp.symbols("M_w rho_w rho_ice sigma_w D_v k_a")
g, mu_air, rho_air, Cd = sp.symbols("g mu_air rho_air Cd")
cv_a, cv_v, c_w, c_i = sp.symbols("cv_a cv_v c_w c_i")
M_s, rho_s, i_vh, kappa = sp.symbols("M_s rho_s i_vh kappa")


# =====================================================================
# 1a. voxel_air_step -- the dry-air carrier of one Eulerian cell
# =====================================================================
# The cell's air is stepped once; each condensable species in the cell is
# then stepped by voxel_species_step riding the SAME face flows F_*.  The
# caller sums the species' Q_lat and C_species and hands them back here as
# Q_ext / C_cond, so any number of species share one temperature.  P_vap is
# the sum of the species' partial pressures e (last tick's, one-step lag).
FACES = ("xm", "xp", "ym", "yp", "zm", "zp")
dx = sp.Symbol("dx")
m_a, P_vap, C_cond, Q_ext = sp.symbols("m_a P_vap C_cond Q_ext")
k_flow, dP_blend = sp.symbols("k_flow dP_blend")
P_n = {f: sp.Symbol(f"P_{f}") for f in FACES}
T_n = {f: sp.Symbol(f"T_{f}") for f in FACES}

V = dx**3
A_face = dx**2
rho_a = m_a / V
P = m_a * R_a * T / V + P_vap                        # ideal-gas mixture, constant volume

# Pressure-driven face exchange; the donor cell is chosen by a smooth step
# on the pressure difference, so enthalpy rides the air without a branch and
# outflow carries this cell's own state.
F_face = {}
dm_a_flow = 0
dQ_flow = 0
dQ_cond = 0
for f in FACES:
    F_face[f] = dt * k_flow * A_face * (P_n[f] - P)
    up = smooth_step(P_n[f] - P, dP_blend)
    T_don = up * T_n[f] + (1 - up) * T
    dm_a_flow += F_face[f]
    dQ_flow += F_face[f] * cv_a * (T_don - T)
    dQ_cond += dt * k_a * dx * (T_n[f] - T)          # Fourier, face area / dx

C_cell = m_a * cv_a + C_cond
alpha_a = k_a / (rho_a * cv_a + EPS)
tau_p = dx / (6 * k_flow * R_a * T + EPS)            # pressure equalisation time

m_a_n = sp.Max(m_a + dm_a_flow, 0)

VOXEL_AIR_STEP = [
    named("m_a_next", m_a_n),
    named("T_next", T + (dQ_flow + dQ_cond + Q_ext) / (C_cell + EPS)),
    named("P", P),
    named("rho_a", rho_a),
    *[named(f"F_{f}", F_face[f]) for f in FACES],
    named("dt_limit", sp.Min(dx**2 / (6 * alpha_a + EPS), tau_p)),
    named("max_vel", sp.Max(*[smooth_abs(F_face[f]) for f in FACES]) / (rho_a * A_face * dt + TINY)),
    named("max_flux", sum(smooth_abs(F_face[f]) for f in FACES) / dt),
    named("div_inf", smooth_abs(dm_a_flow) / ((m_a + TINY) * dt)),
    named("mass_err", smooth_abs(m_a_n - (m_a + dm_a_flow)) / (m_a + TINY)),
    named("energy_j", C_cell * T),
    named("power_w", smooth_abs(dQ_flow + dQ_cond + Q_ext) / dt),
]


# =====================================================================
# 1b. voxel_species_step -- one condensable species in that cell
# =====================================================================
# Four categories per species (Kessler 1969 bulk microphysics): vapour,
# cloud (suspended, settles slowly), ice, rain (falls fast).  Cloud appears
# the instant the cell is cooled or mixed past this species' own dew point
# -- nothing here says "make fog"; rain appears when cloud water exceeds
# the autoconversion threshold and then grows by sweeping cloud (accretion)
# as it falls, so a cloud rains itself out.  Mixing two saturated parcels at
# different temperatures supersaturates the mixture because e_s(T) is
# convex: that is contrail and breath fog, and it falls out of the face
# exchange with no extra rule.
m_v, m_l, m_i, m_r = sp.symbols("m_v m_l m_i m_r")
F_blend, tau_cond, tau_freeze = sp.symbols("F_blend tau_cond tau_freeze")
w_l, w_i, w_r = sp.symbols("w_l w_i w_r")
m_l_up, m_i_up, m_r_up, w_l_up, w_i_up, w_r_up = sp.symbols(
    "m_l_up m_i_up m_r_up w_l_up w_i_up w_r_up")
q_c0, k_auto, k_acc, k_revap = sp.symbols("q_c0 k_auto k_acc k_revap")
F_in = {f: sp.Symbol(f"F_{f}") for f in FACES}      # from voxel_air_step
q_n = {f: sp.Symbol(f"q_{f}") for f in FACES}      # neighbour specific humidity, kg/kg

q = m_v / (m_a + TINY)
e = m_v * R_v * T / V                                # vapour partial pressure
e_s = saturation_pressure(T, P_tp, T_tp, L_v, R_v)
e_si = saturation_pressure(T, P_tp, T_tp, L_v + L_f, R_v)
f_ice = hard_step(T_tp - T, sp.Float(1.0))

dm_v_flow = 0
dm_v_diff = 0
for f in FACES:
    up = smooth_step(F_in[f], F_blend)
    q_don = up * q_n[f] + (1 - up) * q
    dm_v_flow += F_in[f] * q_don
    dm_v_diff += dt * D_v * dx * rho_a * (q_n[f] - q)    # Fick, face area / dx

# Relaxation to saturation over liquid or ice, weighted by the ice regime;
# evaporation can never take more condensate than the cell holds.
m_vs = (f_ice * e_si + (1 - f_ice) * e_s) * V / (R_v * T)
S = e / e_s
dm_cond = sp.Max((m_v - m_vs) * relax(dt, tau_cond), -m_l)
m_l1 = m_l + dm_cond
dm_freeze = (f_ice * (m_l1 + m_i) - m_i) * relax(dt, tau_freeze)

# Warm rain: cloud -> rain past a threshold mixing ratio, rain sweeping
# cloud as it falls, rain evaporating in subsaturated air.
q_c = m_l1 / (m_a + EPS)
q_r = m_r / (m_a + EPS)
auto = k_auto * sp.Max(q_c - q_c0, 0)
accr = k_acc * q_c * sp.Pow(q_r + EPS, sp.Float(0.875))
dm_auto = sp.Min(dt * m_a * (auto + accr), m_l1)
revap = k_revap * sp.Max(1 - S, 0) * sp.Pow(q_r + EPS, sp.Float(0.65))
dm_revap = sp.Min(dt * m_a * revap, m_r)

s_l = sp.Min(w_l * dt / dx, 1)
s_i = sp.Min(w_i * dt / dx, 1)
s_r = sp.Min(w_r * dt / dx, 1)
drizzle_out = m_l * s_l
snow_out = m_i * s_i
rain_out = m_r * s_r
drizzle_in = m_l_up * sp.Min(w_l_up * dt / dx, 1)
snow_in = m_i_up * sp.Min(w_i_up * dt / dx, 1)
rain_in = m_r_up * sp.Min(w_r_up * dt / dx, 1)

Q_lat = L_v * (dm_cond - dm_revap) + L_f * dm_freeze
C_species = m_v * cv_v + (m_l + m_r) * c_w + m_i * c_i

m_v_n = sp.Max(m_v + dm_v_flow + dm_v_diff - dm_cond + dm_revap, 0)
m_l_n = sp.Max(m_l1 - dm_freeze - dm_auto - drizzle_out + drizzle_in, 0)
m_i_n = sp.Max(m_i + dm_freeze - snow_out + snow_in, 0)
m_r_n = sp.Max(m_r + dm_auto - dm_revap - rain_out + rain_in, 0)
total_water = m_v + m_l + m_i + m_r
expected_change = (dm_v_flow + dm_v_diff + drizzle_in + snow_in + rain_in
                   - drizzle_out - snow_out - rain_out)

VOXEL_SPECIES_STEP = [
    named("m_v_next", m_v_n),
    named("m_l_next", m_l_n),
    named("m_i_next", m_i_n),
    named("m_r_next", m_r_n),
    named("e", e),
    named("e_s", e_s),
    named("S", S),
    named("LWC", m_l / V),
    named("IWC", m_i / V),
    named("RWC", m_r / V),
    named("Q_lat", Q_lat),
    named("C_species", C_species),
    named("rain_out", rain_out),
    named("snow_out", snow_out),
    named("drizzle_out", drizzle_out),
    named("cond_rate", dm_cond / dt),
    named("auto_rate", dm_auto / dt),
    named("rain_fraction", dm_auto / (m_l1 + TINY)),
    named("dt_limit", sp.Min(
        dx**2 / (6 * D_v + EPS), tau_cond, tau_freeze,
        dx / (sp.Max(w_l, w_i, w_r) + EPS),
        euler_bound(-m_a * (auto + accr), m_l),
        euler_bound(-m_a * revap, m_r),
    )),
    named("max_vel", sp.Max(w_l, w_i, w_r)),
    named("max_flux", (smooth_abs(dm_v_flow) + smooth_abs(dm_cond) + dm_auto + dm_revap
                       + rain_out + drizzle_out + snow_out) / dt),
    named("div_inf", smooth_abs(dm_v_flow + dm_v_diff) / ((m_v + TINY) * dt)),
    named("mass_err", smooth_abs((m_v_n + m_l_n + m_i_n + m_r_n) - total_water - expected_change)
          / (total_water + TINY)),
    named("energy_j", C_species * T + L_v * m_v),
    named("power_w", smooth_abs(Q_lat) / dt),
]


# =====================================================================
# 1c. aerosol_step -- smoke, dust or salt haze in that cell, as moments
# =====================================================================
N_a, M_a = sp.symbols("N_a M_a")                     # number / m^3, dry mass kg / m^3
rho_p, lam_mfp, k_B, kappa_a = sp.symbols("rho_p lam_mfp k_B kappa_a")
E_N, E_M, dep_in_N, dep_in_M = sp.symbols("E_N E_M dep_in_N dep_in_M")
delta_bl, f_wall, k_wash_r, tau_cond_max = sp.symbols("delta_bl f_wall k_wash_r tau_cond_max")
S_amb_a, LWC_a, RWC_a, rain_fraction = sp.symbols("S_amb LWC RWC rain_fraction")
lam_light, refr_factor = sp.symbols("lam_light refr_factor")
N_n = {f: sp.Symbol(f"N_{f}") for f in FACES}
M_n = {f: sp.Symbol(f"M_{f}") for f in FACES}

# Representative particle: the mean-volume sphere of the population.
v_bar = M_a / (rho_p * N_a + TINY)
D_m = cbrt(6 * v_bar / pi + TINY)
Cc = 1 + 2.514 * lam_mfp / D_m                       # Cunningham slip
K_coag = 8 * k_B * T * Cc / (3 * mu_air)             # Brownian, monodisperse
v_s = rho_p * g * D_m**2 * Cc / (18 * mu_air)        # Stokes settling
D_B = k_B * T * Cc / (3 * pi * mu_air * D_m)         # Brownian diffusivity
v_wall = D_B / delta_bl

dN_coag = sp.Min(HALF * K_coag * N_a**2 * dt, HALF * N_a)
dN_flow = 0
dM_flow = 0
for f in FACES:
    vol = F_in[f] / (rho_a * V)                      # fraction of this cell's volume exchanged
    up = smooth_step(F_in[f], F_blend)
    dN_flow += vol * (up * N_n[f] + (1 - up) * N_a)
    dM_flow += vol * (up * M_n[f] + (1 - up) * M_a)

loss_settle = sp.Min(v_s * dt / dx, 1)
loss_wall = sp.Min(v_wall * 6 * f_wall * dt / dx, 1)
Lambda_wash = k_wash_r * sp.Pow(RWC_a + TINY, sp.Float(0.8))
loss_wash = sp.Min(Lambda_wash * dt, 1)

# kappa-Koehler activation of the mean dry diameter: above S_crit the
# particle has no stable haze radius and becomes a cloud droplet.
A_kelvin_D = 4 * M_w * sigma_w / (R_u * T * rho_w)
S_crit = sp.exp(sp.sqrt(4 * A_kelvin_D**3 / (27 * kappa_a * D_m**3 + TINY)))
f_act = smooth_step(S_amb_a - S_crit, sp.Float(0.002))
loss_rainout = sp.Min(f_act * rain_fraction, 1)
survive = (1 - loss_settle) * (1 - loss_wall) * (1 - loss_wash) * (1 - loss_rainout)

N_pre = N_a + E_N * dt + dep_in_N * dt / V + dN_flow - dN_coag
M_pre = M_a + E_M * dt + dep_in_M * dt / V + dM_flow
N_a_n = sp.Max(N_pre * survive, 0)
M_a_n = sp.Max(M_pre * survive, 0)

N_drop = f_act * N_a
r_drop = cbrt(3 * LWC_a / (4 * pi * rho_w * N_drop + TINY) + TINY)
tau_cond_out = sp.Min(1 / (4 * pi * D_v * N_drop * r_drop + TINY), tau_cond_max)

x_size = pi * D_m / lam_light
mie_w = x_size**4 / (1 + x_size**4)
beta_ext = N_a * (pi / 4) * D_m**2 * (2 * mie_w + sp.Rational(8, 3) * x_size**4 * refr_factor * (1 - mie_w))

AEROSOL_STEP = [
    named("N_next", N_a_n),
    named("M_next", M_a_n),
    named("D_mean", D_m),
    named("v_settle", v_s),
    named("N_drop", N_drop),
    named("f_act", f_act),
    named("S_crit", S_crit),
    named("tau_cond_out", tau_cond_out),
    named("dep_out_N", N_pre * loss_settle * V / dt),
    named("dep_out_M", M_pre * loss_settle * V / dt),
    named("wall_dep_M", M_pre * loss_wall * V / dt),
    named("washout_M", M_pre * loss_wash * V / dt),
    named("beta_ext", beta_ext),
    named("mie_weight", mie_w),
    named("transmittance", sp.exp(-beta_ext * dx)),
    named("dt_limit", sp.Min(
        1 / (K_coag * N_a + EPS),
        dx / (v_s + EPS),
        dx / (6 * f_wall * v_wall + EPS),
        1 / (Lambda_wash + EPS),
    )),
    named("max_vel", sp.Max(v_s, v_wall)),
    named("max_flux", E_M + (dep_in_M + smooth_abs(dM_flow) * V / dt)
          + M_pre * (loss_settle + loss_wall + loss_wash + loss_rainout) * V / dt),
    named("div_inf", smooth_abs(dM_flow) / ((M_a + TINY) * dt)),
    named("mass_err", smooth_abs(M_a_n - M_pre * survive) / (M_a + TINY)),
]


# =====================================================================
# 2. droplet_step -- one Lagrangian drop or crystal carrying one salt
# =====================================================================
r, n_s, T_p, e_amb, z, w, c_s = sp.symbols("r n_s T_p e_amb z w c_s")

r_d = cbrt(3 * n_s * M_s / (4 * pi * rho_s))          # dry crystal radius
r3, rd3 = r**3, r_d**3
a_w_drop = (r3 - rd3) / (r3 - rd3 * (1 - kappa) + TINY)  # kappa-Koehler water activity
A_kelvin = 2 * M_w * sigma_w / (R_u * T_p * rho_w)
S_eq = a_w_drop * sp.exp(A_kelvin / r)
e_s_amb = saturation_pressure(T, P_tp, T_tp, L_v, R_v)
S_amb = e_amb / e_s_amb

# Mason: r dr/dt = (S - S_eq) / (F_k + F_d); stepped on r^2, floored at the
# dry radius so a drying drop becomes its crystal instead of vanishing.
F_k = (L_v / (R_v * T) - 1) * L_v * rho_w / (k_a * T)
F_d = rho_w * R_v * T / (D_v * e_s_amb)
dr2 = sp.Max(2 * dt * (S_amb - S_eq) / (F_k + F_d), r_d**2 - r**2)   # the floored change itself
r2_next = r**2 + dr2
r_next = sp.sqrt(r2_next)

m_w_drop = sp.Rational(4, 3) * pi * rho_w * (r3 - rd3)
m_p = m_w_drop + n_s * M_s
C_p = m_w_drop * c_w + n_s * M_s * c_s + TINY
# r_next^3 - r^3 as a difference of nearly equal cubes cancels to a few
# digits; (r_next - r)(r_next^2 + r_next r + r^2) with r_next - r written
# from the r^2 step itself carries the full mass change.
dr = dr2 / (r_next + r)
dm_w = sp.Rational(4, 3) * pi * rho_w * dr * (r_next**2 + r_next * r + r**2)

# Drop temperature: convective relaxation to the air (Nu = 2) is integrated
# exactly, with the latent heat of this step's mass change as a constant
# source, so no explicit thermal stability limit exists.
h_c = k_a / r
tau_T = C_p / (h_c * 4 * pi * r**2 + TINY)
r_T = relax(dt, tau_T)
T_p_next = T + (T_p - T) * (1 - r_T) + (L_v * dm_w / C_p) * (tau_T / dt) * r_T

rho_p = m_p / (sp.Rational(4, 3) * pi * r3 + TINY)
v_stokes = 2 * rho_p * g * r**2 / (9 * mu_air)
v_newton = sp.sqrt(8 * rho_p * g * r / (3 * Cd * rho_air))
v_t = v_stokes * v_newton / sp.sqrt(v_stokes**2 + v_newton**2 + TINY)
tau_v = v_t / g
w_next = v_t + (w - v_t) * sp.exp(-dt / (tau_v + TINY))

# Explicit Euler on r^2 is stable while dt < 2 F r / |dS_eq/dr|: below the
# Koehler activation supersaturation a haze drop sits at a stable
# equilibrium radius, above it there is none and the drop runs away into a
# cloud drop.  This bound keeps the stable branch stable and leaves the
# runaway alone; the Mason bound keeps one step from overshooting S_eq.
dS_eq_dr = sp.diff(S_eq, r)
F_sum = F_k + F_d

DROPLET_STEP = [
    named("r_next", r_next),
    named("T_p_next", T_p_next),
    named("w_next", w_next),
    named("z_next", z - w_next * dt),
    named("S", S_amb),
    named("S_eq", S_eq),
    named("a_w", a_w_drop),
    named("v_t", v_t),
    named("m_p", m_p),
    named("r_d", r_d),
    named("dry_fraction", smooth_step(sp.Float(1.05) * r_d - r, sp.Float(0.02) * r_d)),
    named("dt_limit", sp.Min(
        r**2 * F_sum / (2 * smooth_abs(S_amb - S_eq) + EPS),
        2 * F_sum * r / (smooth_abs(dS_eq_dr) + EPS),
    )),
    named("max_vel", smooth_abs(w_next)),
    named("max_flux", smooth_abs(dm_w) / dt),
    named("mass_err", smooth_abs(r_next**2 - r2_next) / (r**2 + TINY)),
    named("energy_j", C_p * T_p),
    named("power_w", smooth_abs(L_v * dm_w) / dt + h_c * 4 * pi * r**2 * smooth_abs(T - T_p)),
]


# =====================================================================
# 3. salt_solution_step -- one inorganic salt in one body of water
# =====================================================================
m_w, n_c, b_sat0, db_sat_dT, T_ref = sp.symbols("m_w n_c b_sat0 db_sat_dT T_ref")
K_f, K_b, T_f0, T_b0 = sp.symbols("K_f K_b T_f0 T_b0")
RH, DRH, ERH, h_wet = sp.symbols("RH DRH ERH h_wet")
k_grow, k_diss, k_nuc, B_nuc, tau_w = sp.symbols("k_grow k_diss k_nuc B_nuc tau_w")
dH_diss, c_s_salt, Lambda_m, T_sol = sp.symbols("dH_diss c_s_salt Lambda_m T_sol")

n_diss = sp.Max(n_s - n_c, 0)
b_mol = n_diss / (m_w + EPS)                          # molality, mol / kg water
b_sat = sp.Max(b_sat0 + db_sat_dT * (T - T_ref), EPS)
S_c = b_mol / b_sat
V_c = n_c * M_s / rho_s
A_c = sp.Pow(V_c + EPS, sp.Rational(2, 3))            # crystal surface ~ volume^(2/3)
growth = k_grow * A_c * sp.Max(S_c - 1, 0) ** 2
nucleation = k_nuc * m_w * sp.exp(-B_nuc / sp.log(sp.Max(S_c, 1 + EPS)) ** 2)
dissolution = k_diss * A_c * sp.Max(1 - S_c, 0)
dn_c = dt * (growth + nucleation - dissolution)

a_w_sol = 1 / (1 + i_vh * b_mol * M_w)                # Raoult, molality form
RH_c = sp.Min(RH, sp.Float(0.999))
m_w_eq = h_wet * n_s * i_vh * M_w * RH_c / (1 - RH_c)  # water held at a_w = RH
w_rh = sp.Float(0.005)

n_c_n = sp.Min(sp.Max(n_c + dn_c, 0), n_s)
# Crystallising releases the enthalpy dissolution absorbed (dH_diss > 0 for
# NaCl, < 0 for CaCl2): an endothermic swing is a sign change of Q_reaction.
dn_real = sp.Min(sp.Max(dn_c, -n_c), n_s - n_c)       # the clamped change, not next - now
Q_reaction = dH_diss * dn_real
C_sol = m_w * c_w + n_s * M_s * c_s_salt + TINY
T_sol_n = T_sol + Q_reaction / C_sol
T_freeze = T_f0 - K_f * i_vh * b_mol
sigma_sol = Lambda_m * n_diss / (m_w / rho_w + TINY)             # molar conductivity x molarity

SALT_SOLUTION_STEP = [
    named("T_sol_next", T_sol_n),
    named("Q_reaction", Q_reaction),
    named("endothermic_power", sp.Max(-Q_reaction, 0) / dt),
    named("freeze_margin", T_sol_n - T_freeze),
    named("sigma_sol", sigma_sol),
    named("n_c_next", n_c_n),
    named("m_w_next", sp.Max(m_w + (m_w_eq - m_w) * relax(dt, tau_w), 0)),
    named("h_next", clamp01(h_wet + hard_step(RH - DRH, w_rh) - hard_step(ERH - RH, w_rh))),
    named("b", b_mol),
    named("S_c", S_c),
    named("a_w", a_w_sol),
    named("e_sol", a_w_sol * saturation_pressure(T, P_tp, T_tp, L_v, R_v)),
    named("T_freeze", T_freeze),
    named("T_boil", T_b0 + K_b * i_vh * b_mol),
    named("V_c", V_c),
    named("V_sol", m_w / rho_w + n_diss * M_s / rho_s),
    named("crystal_fraction", n_c / (n_s + EPS)),
    named("dt_limit", sp.Min(
        tau_w + EPS,
        euler_bound(growth + nucleation - dissolution, n_c),
    )),
    named("max_flux", smooth_abs(dn_c) * M_s / dt),
    named("mass_err", smooth_abs(n_c_n - (n_c + dn_c)) / (n_s + EPS)),
    named("energy_j", C_sol * T_sol),
    named("power_w", smooth_abs(Q_reaction) / dt),
]


# =====================================================================
# 4. surface_step -- one bounded face: dew, frost, crust, sediment, runoff
# =====================================================================
A_s, T_s, T_plate, U_plate, h_conv, C_s, alpha_evap = sp.symbols(
    "A_s T_s T_plate U_plate h_conv C_s alpha_evap")
m_film, n_s_film, m_frost, rho_frost, m_crust, m_sed = sp.symbols(
    "m_film n_s_film m_frost rho_frost m_crust m_sed")
phi_drop, s_drop, phi_sed, k_wash = sp.symbols("phi_drop s_drop phi_sed k_wash")
sigma_l, theta_c, mu_l, tilt, w_edge, dcos_hyst = sp.symbols(
    "sigma_l theta_c mu_l tilt w_edge dcos_hyst")
tau_dens, tau_cryst, k_frost, rho_sed = sp.symbols("tau_dens tau_cryst k_frost rho_sed")

f_i = hard_step(T_tp - T_s, sp.Float(0.5))
b_sat_film = sp.Max(b_sat0 + db_sat_dT * (T_s - T_ref), EPS)   # solubility at the surface
b_film = n_s_film / (m_film + EPS)
a_w_film = 1 / (1 + i_vh * b_film * M_w)
P_sat = f_i * saturation_pressure(T_s, P_tp, T_tp, L_v + L_f, R_v) \
    + (1 - f_i) * a_w_film * saturation_pressure(T_s, P_tp, T_tp, L_v, R_v)
J = interface_flux(P_sat, e_amb, T_s, alpha_evap, h_conv, M_w, R_v, rho_air, cv_a + R_a)
dm_total = -J * A_s * dt
dm_liq = sp.Max((1 - f_i) * dm_total, -m_film)
dm_ice = sp.Max(f_i * dm_total, -m_frost)

# Surface tension holds a sessile liquid at h_sessile = 2 l_c sin(theta/2),
# l_c = sqrt(sigma / (rho g)) the capillary length (de Gennes).  Tilted, the
# gravity-normal component shrinks that, and what still clings is the
# pinned-drop height from contact-angle hysteresis, rho g sin(tilt) h^2 / 2
# = sigma (cos theta_r - cos theta_a): a floor never drains (the pool law
# owns it), a wall holds about a millimetre.  Only the excess drains, as a
# Nusselt gravity film of mean speed rho g sin(tilt) h^2 / (3 mu) leaving
# over the edge width w_edge.
l_c = sp.sqrt(sigma_l / (rho_w * g))
h_sessile = 2 * l_c * sp.sin(theta_c / 2)
h_cap = h_sessile * sp.cos(tilt) + l_c * sp.sqrt(2 * dcos_hyst / (sp.sin(tilt) + EPS))
h_film = m_film / (A_s * rho_w)
h_excess = sp.Max(h_film - h_cap, 0)
u_film = rho_w * g * sp.sin(tilt) * h_excess**2 / (3 * mu_l)
runoff = sp.Min(rho_w * u_film * h_excess * w_edge * dt, m_film)
f_run = runoff / (m_film + EPS)
m_film_1 = sp.Max(m_film + dm_liq + phi_drop * dt - runoff, 0)

n_after = n_s_film * (1 - f_run) + phi_drop * s_drop * dt
n_excess = sp.Max(n_after - b_sat_film * m_film_1, 0)     # beyond what the water can hold
dn_cryst = n_excess * relax(dt, tau_cryst)
n_deficit = sp.Max(b_sat_film * m_film_1 - n_after, 0)
dn_diss = sp.Min(m_crust / M_s, n_deficit) * relax(dt, tau_cryst)

h_frost = m_frost / (A_s * rho_frost + EPS)
U_eff = 1 / (1 / U_plate + h_frost / k_frost)
m_sed_next = sp.Max(m_sed + phi_sed * dt - m_sed * k_wash * f_run, 0)

Q_lat = dm_liq * L_v + dm_ice * (L_v + L_f)          # deposition heats the surface
Q_conv_s = h_conv * A_s * (T - T_s) * dt
Q_plate = U_eff * A_s * (T_plate - T_s) * dt
C_tot = C_s * A_s + m_film * c_w + m_frost * c_i

m_frost_n = sp.Max(m_frost + dm_ice, 0)

SURFACE_STEP = [
    named("m_film_next", m_film_1),
    named("h_film", h_film),
    named("h_cap", h_cap),
    named("m_frost_next", m_frost_n),
    named("rho_frost_next", rho_frost + (rho_ice - rho_frost) * relax(dt, tau_dens)),
    named("h_frost", h_frost),
    named("m_crust_next", sp.Max(m_crust + (dn_cryst - dn_diss) * M_s, 0)),
    named("n_s_next", sp.Max(n_after - dn_cryst + dn_diss, 0)),
    named("m_sed_next", m_sed_next),
    named("T_s_next", T_s + (Q_lat + Q_conv_s + Q_plate) / (C_tot + EPS)),
    named("J", J),
    named("Q_gas", -Q_conv_s),
    named("runoff", runoff),
    named("a_w_film", a_w_film),
    named("U_eff", U_eff),
    named("crust_thickness", m_crust / (rho_s * A_s)),
    named("sediment_thickness", m_sed / (rho_sed * A_s)),
    named("dt_limit", sp.Min(
        (C_tot + EPS) / ((h_conv + U_eff) * A_s + EPS),
        euler_bound(-(1 - f_i) * J * A_s - rho_w * u_film * h_excess * w_edge, m_film),
        tau_dens + EPS, tau_cryst + EPS,
    )),
    named("max_vel", u_film),
    named("max_flux", smooth_abs(J) * A_s + phi_drop + phi_sed),
    named("mass_err", (smooth_abs(m_film_1 - (m_film + dm_liq + phi_drop * dt - runoff))
                       + smooth_abs(m_frost_n - (m_frost + dm_ice))) / (m_film + m_frost + TINY)),
    named("energy_j", C_tot * T_s),
    named("power_w", smooth_abs(Q_lat + Q_conv_s + Q_plate) / dt),
]


# =====================================================================
# 5. pool_step -- one liquid body on a floor
# =====================================================================
m_p, n_s_pool, T_l, A_floor = sp.symbols("m_p n_s_pool T_l A_floor")
phi_in, s_in, T_in = sp.symbols("phi_in s_in T_in")
h_sill, w_sill, C_w, T_floor, U_floor = sp.symbols("h_sill w_sill C_w T_floor U_floor")

# Geometry is derived: a puddle spreads at the sessile thickness until it
# covers the floor, then deepens.  A sill drains it as a sharp-crested weir,
# Q = C_w w sqrt(2 g) H^(3/2).
V_l = m_p / rho_w
A_wet = sp.Min(V_l / (h_sessile + TINY), A_floor)
h_pool = V_l / (A_wet + TINY)
b_pool = n_s_pool / (m_p + EPS)
a_w_pool = 1 / (1 + i_vh * b_pool * M_w)
J_pool = interface_flux(a_w_pool * saturation_pressure(T_l, P_tp, T_tp, L_v, R_v), e_amb,
                        T_l, alpha_evap, h_conv, M_w, R_v, rho_air, cv_a + R_a)
dm_evap = sp.Min(J_pool * A_wet * dt, m_p)                 # negative = condensing onto the pool
H_sill = sp.Max(h_pool - h_sill, 0)
overflow = sp.Min(rho_w * C_w * w_sill * sp.sqrt(2 * g) * sp.Pow(H_sill, sp.Rational(3, 2)) * dt, m_p)
m_p_n = sp.Max(m_p + phi_in * dt - dm_evap - overflow, 0)
f_over = overflow / (m_p + EPS)

C_l = m_p * c_w + TINY
Q_lat_pool = -dm_evap * L_v
Q_conv_pool = h_conv * A_wet * (T - T_l) * dt
Q_floor = U_floor * A_wet * (T_floor - T_l) * dt
Q_in = phi_in * dt * c_w * (T_in - T_l)
pool_mass_rate = phi_in - J_pool * A_wet - overflow / dt

POOL_STEP = [
    named("m_p_next", m_p_n),
    named("n_s_next", sp.Max(n_s_pool * (1 - f_over) + phi_in * s_in * dt, 0)),
    named("T_l_next", T_l + (Q_lat_pool + Q_conv_pool + Q_floor + Q_in) / C_l),
    named("V_l", V_l),
    named("A_wet", A_wet),
    named("h_pool", h_pool),
    named("h_cap", h_sessile),
    named("b", b_pool),
    named("a_w", a_w_pool),
    named("J", J_pool),
    named("Q_gas", -Q_conv_pool),
    named("evap_rate", dm_evap / dt),
    named("overflow_rate", overflow / dt),
    named("dt_limit", sp.Min(
        C_l / ((h_conv + U_floor) * A_wet + phi_in * c_w + EPS),
        euler_bound(pool_mass_rate, m_p),
    )),
    named("max_vel", C_w * sp.sqrt(2 * g * H_sill)),
    named("max_flux", phi_in + smooth_abs(J_pool) * A_wet + overflow / dt),
    named("mass_err", smooth_abs(m_p_n - (m_p + phi_in * dt - dm_evap - overflow)) / (m_p + TINY)),
    named("energy_j", C_l * T_l),
    named("power_w", smooth_abs(Q_lat_pool + Q_conv_pool + Q_floor + Q_in) / dt),
]


# =====================================================================
# 6. Compilation API for tools/compile_symbolic_source.py
# =====================================================================
LAWS = {
    "voxel_air_step": VOXEL_AIR_STEP,
    "voxel_species_step": VOXEL_SPECIES_STEP,
    "aerosol_step": AEROSOL_STEP,
    "droplet_step": DROPLET_STEP,
    "salt_solution_step": SALT_SOLUTION_STEP,
    "surface_step": SURFACE_STEP,
    "pool_step": POOL_STEP,
}

LAW_PUBLICATIONS = {
    "voxel_air_step": (
        SymbolicPublication(output="P", semantic="pressure", unit="Pa"),
        SymbolicPublication(output="T_next", semantic="temperature", unit="K"),
    ),
    "voxel_species_step": (
        SymbolicPublication(output="S", semantic="saturation_ratio", unit="1"),
        SymbolicPublication(output="LWC", semantic="cloud_water_content", unit="kg/m^3"),
        SymbolicPublication(output="IWC", semantic="ice_water_content", unit="kg/m^3"),
        SymbolicPublication(output="RWC", semantic="rain_water_content", unit="kg/m^3"),
    ),
    "aerosol_step": (
        SymbolicPublication(output="beta_ext", semantic="extinction_coefficient", unit="1/m"),
        SymbolicPublication(output="mie_weight", semantic="mie_rayleigh_blend", unit="1"),
        SymbolicPublication(output="N_drop", semantic="activated_droplet_number", unit="1/m^3"),
    ),
    "droplet_step": (
        SymbolicPublication(output="r_next", semantic="wet_radius", unit="m"),
        SymbolicPublication(output="S_eq", semantic="koehler_equilibrium_saturation", unit="1"),
        SymbolicPublication(output="dry_fraction", semantic="crystal_state", unit="1"),
    ),
    "salt_solution_step": (
        SymbolicPublication(output="S_c", semantic="solution_supersaturation", unit="1"),
        SymbolicPublication(output="crystal_fraction", semantic="precipitated_fraction", unit="1"),
        SymbolicPublication(output="T_freeze", semantic="freezing_point", unit="K"),
    ),
    "surface_step": (
        SymbolicPublication(output="h_film", semantic="dew_film_thickness", unit="m"),
        SymbolicPublication(output="h_frost", semantic="frost_thickness", unit="m"),
        SymbolicPublication(output="crust_thickness", semantic="salt_crust_thickness", unit="m"),
        SymbolicPublication(output="sediment_thickness", semantic="sediment_thickness", unit="m"),
    ),
    "pool_step": (
        SymbolicPublication(output="A_wet", semantic="wetted_area", unit="m^2"),
        SymbolicPublication(output="h_pool", semantic="pool_depth", unit="m"),
        SymbolicPublication(output="overflow_rate", semantic="weir_overflow", unit="kg/s"),
    ),
}

DTYPE = "float64"
SCHEDULE = "asap"
BATCH = 1
