"""Electromagnetics, waves and charged species as simultaneous SymPy laws.

Same contract and conventions as symbolic_chamber_solvers.py: one call is one
explicit transition on one cell (or one particle), outputs are fresh names,
neighbour values arrive as inputs gathered by the package, every law
publishes ``dt_limit`` from its own parameters and the dt system's metric
names.  These laws are the field side of a controlled chamber: a discharge,
an electrostatic precipitator, an ionic wind, a resonant cavity, a probe
beam through the fog -- and the chemistry side supplies them ``rho_q``,
``J``, ``sigma`` and ``eps_r`` from what is actually in the cell.

LAWS
  maxwell_faraday_step
  maxwell_ampere_step Faraday and Ampere-Maxwell on the Yee lattice, one
                      half-tick each: E on cell edges, H on faces, curls as
                      one-sided differences to the +/- neighbours.  The two
                      are separate laws because Ampere needs H at t + dt/2
                      from the NEIGHBOUR, i.e. after Faraday has run there
                      and the package has gathered it.  Conduction is
                      semi-implicit (stable for any sigma); the source
                      current J comes from the charged species.  Ampere
                      publishes field energy, Poynting power, Joule heat for
                      the gas, local c and impedance, and the Courant limit
                      dx / (c sqrt 3).  At dt = dx / c the pair is exact for
                      a 1D plane wave -- the validation uses that.
  wave_step           The scalar d'Alembert equation psi_tt = c^2 lap psi -
                      gamma psi_t + s (pressure / acoustics in the chamber,
                      c = sqrt(gamma_ad R T) from the air state; or any
                      scalar field), leapfrog in time, 7-point Laplacian.
                      Publishes energy, a local wavenumber and the frequency
                      it implies through the dispersion relation.
  schrodinger_step    A real / imaginary wave function under -hbar^2/2m lap +
                      V, Visscher's staggered leapfrog (the explicit scheme
                      that conserves norm).  Publishes probability density,
                      probability current, energy expectation and the
                      stability limit hbar / (3 hbar^2 / (m dx^2) + |V|).
  charged_species_step
                      One ionic or electron species: number density under
                      Nernst-Planck drift-diffusion on the six faces (donor
                      cell for drift, no branch), production and
                      recombination, wall loss.  Publishes charge density
                      and current density for Maxwell / Poisson, conductivity
                      and permittivity contributions, plasma frequency and
                      Debye length, and the drift / diffusion / dielectric
                      relaxation limits.
  poisson_relax_step  One Jacobi sweep of lap phi = -rho / eps with E =
                      -grad phi, for electrostatic scenarios that do not
                      need the full wave dynamics; ``residual`` is the
                      convergence measure the caller iterates on.

SPECTRAL
  A probe field at wavelength lambda sees the medium through ``eps_r`` and
  ``sigma`` (charged species) and ``beta_ext`` (aerosol / cloud, in the
  chamber laws); the Maxwell pair advances it exactly, and the wave and
  charged-species laws publish the dispersion, plasma-cutoff and skin-depth
  numbers a spectral read-out is made of.
"""

from __future__ import annotations

import sympy as sp

from src.compiler.symbolic_equation_compiler import SymbolicPublication

pi = sp.pi
EPS = sp.Float(1e-12)
TINY = sp.Float(1e-40)
FACES = ("xm", "xp", "ym", "yp", "zm", "zp")
AXES = ("x", "y", "z")


def smooth_abs(x):
    return sp.sqrt(x * x + TINY)


def smooth_step(x, width):
    return sp.Rational(1, 2) * (1 + x / sp.sqrt(x * x + width * width))


def named(name, expr):
    return sp.Eq(sp.Symbol(name), expr, evaluate=False)


def sym(name):
    return sp.Symbol(name)


dt, dx = sp.symbols("dt dx")
V = dx**3


# =====================================================================
# 1. maxwell_yee_step
# =====================================================================
# Yee lattice: Ex lives on the x-edge of the cell, Hx on the x-face.  The
# curl that advances H uses forward differences (the +neighbour), the curl
# that advances E uses backward differences (the -neighbour); with the two
# fields half a step apart in time this is the leapfrog FDTD scheme.
E = {a: sym(f"E{a}") for a in AXES}
H = {a: sym(f"H{a}") for a in AXES}
J = {a: sym(f"J{a}") for a in AXES}
eps, mu, sigma = sp.symbols("eps mu sigma")
Ep = {(a, f): sym(f"E{a}_{f}") for a in AXES for f in ("xp", "yp", "zp")}
Hm = {(a, f): sym(f"H{a}_{f}") for a in AXES for f in ("xm", "ym", "zm")}

curlE = {
    "x": (Ep[("z", "yp")] - E["z"]) / dx - (Ep[("y", "zp")] - E["y"]) / dx,
    "y": (Ep[("x", "zp")] - E["x"]) / dx - (Ep[("z", "xp")] - E["z"]) / dx,
    "z": (Ep[("y", "xp")] - E["y"]) / dx - (Ep[("x", "yp")] - E["x"]) / dx,
}
curlH = {
    "x": (H["z"] - Hm[("z", "ym")]) / dx - (H["y"] - Hm[("y", "zm")]) / dx,
    "y": (H["x"] - Hm[("x", "zm")]) / dx - (H["z"] - Hm[("z", "xm")]) / dx,
    "z": (H["y"] - Hm[("y", "xm")]) / dx - (H["x"] - Hm[("x", "ym")]) / dx,
}
loss = sigma * dt / (2 * eps)
H_n = {a: H[a] - dt / mu * curlE[a] for a in AXES}                 # H(t-dt/2) -> H(t+dt/2)
E_n = {a: (E[a] * (1 - loss) + dt / eps * (curlH[a] - J[a])) / (1 + loss) for a in AXES}  # E(t) -> E(t+dt)

E2 = sum(E[a] ** 2 for a in AXES)
H2 = sum(H[a] ** 2 for a in AXES)
u_field = sp.Rational(1, 2) * (eps * E2 + mu * H2)
poynting = {
    "x": E["y"] * H["z"] - E["z"] * H["y"],
    "y": E["z"] * H["x"] - E["x"] * H["z"],
    "z": E["x"] * H["y"] - E["y"] * H["x"],
}
c_local = 1 / sp.sqrt(eps * mu)
S_mag = sp.sqrt(sum(poynting[a] ** 2 for a in AXES) + TINY)

MAXWELL_FARADAY_STEP = [
    *[named(f"H{a}_next", H_n[a]) for a in AXES],
    named("dt_limit", dx / (c_local * sp.sqrt(3))),
    named("max_vel", c_local),
    named("max_flux", sp.sqrt(sum(curlE[a] ** 2 for a in AXES) + TINY) / mu),
    named("div_inf", sp.Float(0.0)),
    named("mass_err", sp.Float(0.0)),
    named("energy_j", sp.Rational(1, 2) * mu * H2 * V),
    named("power_w", sp.Float(0.0)),
]

MAXWELL_AMPERE_STEP = [
    *[named(f"E{a}_next", E_n[a]) for a in AXES],
    named("u_field", u_field),
    *[named(f"S{a}", poynting[a]) for a in AXES],
    named("Q_joule", sigma * E2 * V * dt),           # heat handed to the gas this step
    named("c_local", c_local),
    named("impedance", sp.sqrt(mu / eps)),
    named("tau_dielectric", eps / (sigma + TINY)),
    named("dt_limit", dx / (c_local * sp.sqrt(3))),
    named("max_vel", c_local),
    named("max_flux", S_mag),
    named("div_inf", smooth_abs(sum(curlH[a] for a in AXES)) / (sp.sqrt(H2) / dx + TINY)),
    named("mass_err", sp.Float(0.0)),
    named("energy_j", u_field * V),
    named("power_w", S_mag * dx**2 + sigma * E2 * V),
]


# =====================================================================
# 2. wave_step -- scalar d'Alembert, leapfrog
# =====================================================================
psi, psi_prev, c_w, gamma_d, s_src = sp.symbols("psi psi_prev c_w gamma_d s_src")
psi_n = {f: sym(f"psi_{f}") for f in FACES}
lap_psi = (sum(psi_n.values()) - 6 * psi) / dx**2
psi_t = (psi - psi_prev) / dt
psi_next = 2 * psi - psi_prev + c_w**2 * dt**2 * lap_psi - gamma_d * dt * (psi - psi_prev) + s_src * dt**2
grad2 = sum(((psi_n[p] - psi_n[m]) / (2 * dx)) ** 2 for p, m in (("xp", "xm"), ("yp", "ym"), ("zp", "zm")))
k_local = sp.sqrt(smooth_abs(lap_psi) / (smooth_abs(psi) + TINY))

WAVE_STEP = [
    named("psi_next", psi_next),
    named("psi_prev_next", psi),
    named("psi_t", psi_t),
    named("lap_psi", lap_psi),
    named("k_local", k_local),
    named("omega_local", c_w * k_local),             # dispersion relation omega = c k
    named("dt_limit", sp.Min(dx / (c_w * sp.sqrt(3)), 2 / (gamma_d + EPS))),
    named("max_vel", c_w),
    named("max_flux", smooth_abs(psi_t)),
    named("div_inf", smooth_abs(lap_psi) * dx**2 / (smooth_abs(psi) + TINY)),
    named("mass_err", sp.Float(0.0)),
    named("energy_j", sp.Rational(1, 2) * (psi_t**2 + c_w**2 * grad2) * V),
    named("power_w", smooth_abs(gamma_d * psi_t**2 - s_src * psi_t) * V),
]


# =====================================================================
# 3. schrodinger_step -- Visscher staggered leapfrog on (R, I)
# =====================================================================
# R is known at integer steps, I at half steps: R_next = R + dt/hbar H I,
# then I_next = I - dt/hbar H R_next.  Norm R^2 + I I' is conserved and the
# scheme is stable for dt < hbar / max|H|.
R_, I_, hbar, m_q, V_pot = sp.symbols("R I_im hbar m_q V_pot")
R_n = {f: sym(f"R_{f}") for f in FACES}
I_n = {f: sym(f"I_{f}") for f in FACES}
kin = hbar**2 / (2 * m_q * dx**2)


def hamiltonian(field, neigh):
    return -kin * (sum(neigh.values()) - 6 * field) + V_pot * field


H_I = hamiltonian(I_, I_n)
R_next = R_ + dt / hbar * H_I
H_Rnext = -kin * (sum(R_n.values()) - 6 * R_next) + V_pot * R_next
I_next = I_ - dt / hbar * H_Rnext
prob = R_ * R_ + I_ * I_
prob_current = {
    a: hbar / m_q * (R_ * (I_n[p] - I_n[m]) - I_ * (R_n[p] - R_n[m])) / (2 * dx)
    for a, (p, m) in zip(AXES, (("xp", "xm"), ("yp", "ym"), ("zp", "zm")))
}
E_max = 6 * kin + smooth_abs(V_pot)

SCHRODINGER_STEP = [
    named("R_next", R_next),
    named("I_next", I_next),
    named("prob", prob),
    *[named(f"j{a}", prob_current[a]) for a in AXES],
    named("energy_density", R_ * hamiltonian(R_, R_n) + I_ * H_I),
    named("dt_limit", hbar / (E_max + TINY)),
    named("max_vel", sp.sqrt(sum(prob_current[a] ** 2 for a in AXES) + TINY) / (prob + TINY)),
    named("max_flux", sp.sqrt(sum(prob_current[a] ** 2 for a in AXES) + TINY)),
    named("div_inf", sp.Float(0.0)),
    named("mass_err", smooth_abs((R_next**2 + I_next**2) - prob) / (prob + TINY)),
    named("energy_j", (R_ * hamiltonian(R_, R_n) + I_ * H_I) * V),
    named("power_w", sp.Float(0.0)),
]


# =====================================================================
# 4. charged_species_step -- Nernst-Planck drift-diffusion
# =====================================================================
n_q, z_q, mu_q, D_q, e_ch, m_ion, T_g = sp.symbols("n_q z_q mu_q D_q e_ch m_ion T_g")
G_ion, k_rec, n_other, k_wall, f_wall, eps_r_per = sp.symbols("G_ion k_rec n_other k_wall f_wall eps_r_per")
A_T, B_T, p_gas, E_bd = sp.symbols("A_T B_T p_gas E_bd")
nn = {f: sym(f"n_{f}") for f in FACES}
Ef = {f: sym(f"E_{f}") for f in FACES}          # field component normal to each face, outward positive
k_B = sp.Symbol("k_B")

# Face flux, outward positive: diffusion down the gradient plus drift of the
# donor cell's density along the outward field.
flux_out = 0
for f in FACES:
    drift_v = z_q * mu_q * Ef[f]                                   # outward drift speed
    up = smooth_step(drift_v, sp.Float(1e-9))
    n_don = up * n_q + (1 - up) * nn[f]
    flux_out += D_q * (n_q - nn[f]) / dx + drift_v * n_don
E_mag = sp.sqrt(sum(Ef[f] ** 2 for f in FACES) / 2 + TINY)
# Townsend: each carrier drifting through the gas ionises alpha_T per metre,
# alpha_T = A p exp(-B p / E).  Below E_bd this is nothing; above it the
# density grows exponentially -- the avalanche a breakdown is made of.
alpha_T = A_T * p_gas * sp.exp(-B_T * p_gas / (E_mag + TINY))
avalanche = alpha_T * smooth_abs(z_q * mu_q) * E_mag * n_q
dn = dt * (G_ion + avalanche - k_rec * n_q * n_other - k_wall * f_wall * n_q - flux_out / dx)
n_next = sp.Max(n_q + dn, 0)
sigma_q = e_ch**2 * z_q**2 * n_q * mu_q
omega_p = sp.sqrt(n_q * e_ch**2 * z_q**2 / (eps * m_ion + TINY))
debye = sp.sqrt(eps * k_B * T_g / (n_q * e_ch**2 * z_q**2 + TINY))
E_axis = {a: (Ef[p] - Ef[m]) / 2 for a, (p, m) in zip(AXES, (("xp", "xm"), ("yp", "ym"), ("zp", "zm")))}

CHARGED_SPECIES_STEP = [
    named("n_next", n_next),
    named("rho_q", e_ch * z_q * n_q),
    *[named(f"J{a}", e_ch * z_q * n_q * z_q * mu_q * E_axis[a]) for a in AXES],
    named("sigma_q", sigma_q),
    named("eps_r_q", eps_r_per * n_q),
    named("omega_p", omega_p),
    named("debye", debye),
    named("skin_depth", sp.sqrt(2 / (mu * sigma_q * omega_p + TINY))),
    named("alpha_townsend", alpha_T),
    named("avalanche_rate", avalanche),
    named("breakdown_margin", E_mag / E_bd),
    named("dt_limit", sp.Min(
        dx**2 / (6 * D_q + EPS),
        dx / (smooth_abs(z_q * mu_q) * E_mag + EPS),
        eps / (sigma_q + EPS),
        1 / (k_rec * n_other + k_wall * f_wall + EPS),
        1 / (alpha_T * smooth_abs(z_q * mu_q) * E_mag + EPS),
    )),
    named("max_vel", smooth_abs(z_q * mu_q) * E_mag),
    named("max_flux", smooth_abs(flux_out) * dx**2),
    named("div_inf", smooth_abs(flux_out) / (dx * (n_q + TINY))),
    named("mass_err", smooth_abs(n_next - (n_q + dn)) / (n_q + TINY)),
    named("energy_j", n_q * V * sp.Rational(3, 2) * k_B * T_g),
    named("power_w", sigma_q * E_mag**2 * V),
]


# =====================================================================
# 5. poisson_relax_step -- one Jacobi sweep of lap phi = -rho / eps
# =====================================================================
phi, rho_tot = sp.symbols("phi rho_tot")
phin = {f: sym(f"phi_{f}") for f in FACES}
phi_next = (sum(phin.values()) + dx**2 * rho_tot / eps) / 6
lap_phi = (sum(phin.values()) - 6 * phi) / dx**2
E_static = {a: -(phin[p] - phin[m]) / (2 * dx) for a, (p, m) in zip(AXES, (("xp", "xm"), ("yp", "ym"), ("zp", "zm")))}

POISSON_RELAX_STEP = [
    named("phi_next", phi_next),
    *[named(f"E{a}", E_static[a]) for a in AXES],
    named("residual", smooth_abs(lap_phi + rho_tot / eps)),
    named("dt_limit", sp.Float(1e30)),               # a relaxation sweep, not a time step
    named("max_vel", sp.Float(0.0)),
    named("max_flux", sp.Float(0.0)),
    named("div_inf", smooth_abs(lap_phi + rho_tot / eps) * dx**2 / (smooth_abs(phi) + TINY)),
    named("mass_err", sp.Float(0.0)),
    named("energy_j", sp.Rational(1, 2) * eps * sum(E_static[a] ** 2 for a in AXES) * V),
    named("power_w", sp.Float(0.0)),
]


# =====================================================================
# 6. Compilation API
# =====================================================================
LAWS = {
    "maxwell_faraday_step": MAXWELL_FARADAY_STEP,
    "maxwell_ampere_step": MAXWELL_AMPERE_STEP,
    "wave_step": WAVE_STEP,
    "schrodinger_step": SCHRODINGER_STEP,
    "charged_species_step": CHARGED_SPECIES_STEP,
    "poisson_relax_step": POISSON_RELAX_STEP,
}

LAW_PUBLICATIONS = {
    "maxwell_ampere_step": (
        SymbolicPublication(output="u_field", semantic="em_energy_density", unit="J/m^3"),
        SymbolicPublication(output="Q_joule", semantic="joule_heat", unit="J"),
    ),
    "wave_step": (
        SymbolicPublication(output="psi_next", semantic="wave_field", unit="1"),
        SymbolicPublication(output="omega_local", semantic="local_angular_frequency", unit="rad/s"),
    ),
    "schrodinger_step": (
        SymbolicPublication(output="prob", semantic="probability_density", unit="1/m^3"),
    ),
    "charged_species_step": (
        SymbolicPublication(output="rho_q", semantic="charge_density", unit="C/m^3"),
        SymbolicPublication(output="omega_p", semantic="plasma_frequency", unit="rad/s"),
    ),
    "poisson_relax_step": (
        SymbolicPublication(output="residual", semantic="poisson_residual", unit="V/m^2"),
    ),
}

DTYPE = "float64"
SCHEDULE = "asap"
BATCH = 1
