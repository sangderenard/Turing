"""Atmospheric moisture/aerosol physics as SymPy equations.

Covers phase state and saturation, dew point condensation, cloud droplet
activation (Koehler curve), sublimation/deposition mass flux, and the
shader-facing extinction/scattering signals used for fog, mist, and cloud
rendering. Kept as plain SymPy so the equations can feed the same
SymPy -> repository SSA -> LLVM compilation path as the other symbolic
physics models in this directory.
"""

from __future__ import annotations

import sympy as sp

pi = sp.pi

# =====================================================================
# 1. Phase State & Saturation (Triple Point Data)
# =====================================================================
T, P_tp, L_v, R_v, T_tp = sp.symbols('T P_tp L_v R_v T_tp')
A, B, C, P = sp.symbols('A B C P')

# Clausius-Clapeyron Relation.
#
# The exponent is (T - T_tp) / (T T_tp), NOT 1/T_tp - 1/T.  They are the
# same number in exact arithmetic and they are not the same number in
# float64: near ambient the two reciprocals agree to several digits and
# their difference cancels those digits away, after which L_v / R_v ~ 5400
# amplifies what is left.  Measured at 295 K, the cancelling spelling is
# 9 ULP off in e_s, and a relaxation to saturation integrates that error
# into real condensate -- it was observed producing cloud water in a cell
# that was exactly saturated and should have produced none.
# symbolic_chamber_solvers.saturation_pressure is spelled this way for the
# same reason; the two files must agree bit for bit, because state seeded
# from one and stepped by the other differs by exactly this.
e_s_expr = P_tp * sp.exp((L_v / R_v) * (T - T_tp) / (T * T_tp))

# Antoine Equation (represented as an equality)
antoine_eq = sp.Eq(sp.log(P, 10), A - B / (C + T))


# =====================================================================
# 2. Vapor Conditions & Condensation (Dew Point)
# =====================================================================
RH, e, e_s, b, c = sp.symbols('RH e e_s b c')

# Relative Humidity Equation
RH_eq = sp.Eq(RH, e / e_s)

# Dew Point Temperature (Magnus-Tetens formula)
gamma = sp.ln(RH) + (b * T) / (c + T)
T_d_expr = (c * gamma) / (b - gamma)


# =====================================================================
# 3. Aerosols and Inorganic Ions (Cloud Droplet Activation)
# =====================================================================
D, M_w, sigma_w, R, rho_w, n_s = sp.symbols('D M_w sigma_w R rho_w n_s')

# Kelvin (curvature) term
A_k = (4 * M_w * sigma_w) / (R * T * rho_w)

# Raoult (solute/ion) term
B_k = (6 * n_s * M_w) / (pi * rho_w)

# Koehler Curve Supersaturation (S)
S_expr = sp.exp(A_k / D - B_k / D**3)


# =====================================================================
# 4. Solid Emitters & Depositions (Sublimation/Condensation)
# =====================================================================
alpha, P_vapor, P_sat_surf, M, T_surface = sp.symbols('alpha P_vapor P_sat_surf M T_surface')

# Hertz-Knudsen Equation for mass flux (J)
J_expr = alpha * (P_vapor - P_sat_surf) / sp.sqrt(2 * pi * M * R * T_surface)


# =====================================================================
# 5. Shader Signals for Fog, Mist, and Clouds
# =====================================================================
k, LWC, d, lamda = sp.symbols('k LWC d lamda')  # 'lamda' avoids Python's lambda keyword

# Extinction Coefficient (beta)
beta_expr = k * LWC

# Beer-Lambert Law (Optical Transmittance)
T_trans_expr = sp.exp(-beta_expr * d)

# Scattering Phase Function Signal (Size Parameter x)
x_expr = (pi * D) / lamda


# =====================================================================
# 6. Named simultaneous equations for compile_sympy_equations()
# =====================================================================
# compile_sympy_equations() requires: each equation's LHS a plain Symbol,
# unique output names, and no output symbol may appear on the RHS of any
# equation in the same call (outputs are simultaneous, not staged). RH_eq
# and T_d_expr are staged diagnostics that depend on e_s and RH as
# intermediate values, so their formulas are substituted in place here
# rather than exposed as separate chained outputs -- the physics is
# unchanged, only which intermediate values get their own output name.
from src.compiler.symbolic_equation_compiler import SymbolicPublication  # noqa: E402

_rh_direct = e / e_s_expr
_t_d_direct = T_d_expr.subs(RH, _rh_direct)

EQUATIONS = [
    sp.Eq(sp.Symbol("e_s"), e_s_expr, evaluate=False),
    sp.Eq(P, sp.Pow(10, A - B / (C + T)), evaluate=False),
    sp.Eq(RH, _rh_direct, evaluate=False),
    sp.Eq(sp.Symbol("T_d"), _t_d_direct, evaluate=False),
    sp.Eq(sp.Symbol("S_koehler"), S_expr, evaluate=False),
    sp.Eq(sp.Symbol("J_flux"), J_expr, evaluate=False),
    sp.Eq(sp.Symbol("beta_ext"), beta_expr, evaluate=False),
    sp.Eq(sp.Symbol("T_trans"), T_trans_expr, evaluate=False),
    sp.Eq(sp.Symbol("x_size"), x_expr, evaluate=False),
]

NAME = "symbolic_atmosphere_model"
FUNCTION_NAME = "symbolic_atmosphere_model"
SCHEDULE = "asap"
DTYPE = "float64"
PUBLICATIONS = (
    SymbolicPublication(output="e_s", semantic="saturation_vapor_pressure", unit="Pa"),
    SymbolicPublication(output="RH", semantic="relative_humidity", unit="1"),
    SymbolicPublication(output="T_d", semantic="dew_point_temperature", unit="K"),
    SymbolicPublication(output="S_koehler", semantic="koehler_supersaturation", unit="1"),
    SymbolicPublication(output="J_flux", semantic="hertz_knudsen_mass_flux", unit="kg/(m^2 s)"),
    SymbolicPublication(output="T_trans", semantic="beer_lambert_transmittance", unit="1"),
)
