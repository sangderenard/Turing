#!/usr/bin/env python3
"""
====================================================
| Orbital.py                                       |
| Canonical symbolic orbital mechanics library     |
| ------------------------------------------------ |
| Provides only pure symbolic textbook equations. |
| For theoretical analysis, program generation,   |
| or direct equivalence to standard references.   |
====================================================
"""

import sympy as sp

# -------------------------------------------------
# Fundamental symbols used throughout
# -------------------------------------------------
t = sp.Symbol('t', real=True)
θ = sp.Symbol('θ', real=True)
μ = sp.Symbol('μ', positive=True)   # gravitational parameter: G(M+m)
a = sp.Symbol('a', positive=True)   # semi-major axis
e = sp.Symbol('e', positive=True)   # eccentricity
r = sp.Symbol('r', positive=True)   # radial distance
E = sp.Symbol('E', real=True)       # eccentric anomaly
M = sp.Symbol('M', real=True)       # mean anomaly

# -------------------------------------------------
# Classical Orbital Mechanics
# -------------------------------------------------

class Orbit:
    """
    Represents a pure symbolic orbit under classical mechanics.
    """

    @staticmethod
    def vis_viva(μ, a, r):
        """
        Vis-viva equation (conservation of energy in orbit):
            v^2 = μ (2/r - 1/a)
        """
        v = sp.Symbol('v', real=True)
        return sp.Eq(v**2, μ * (2/r - 1/a))

    @staticmethod
    def specific_orbital_energy(μ, a):
        """
        Specific orbital energy:
            ε = - μ / (2a)
        """
        ε = sp.Symbol('ε', real=True)
        return sp.Eq(ε, - μ / (2*a))

    @staticmethod
    def orbital_radius_theta(a, e, θ):
        """
        Orbital equation in polar coordinates:
            r(θ) = a(1 - e^2) / (1 + e cos θ)
        """
        return sp.Eq(r, a*(1 - e**2) / (1 + e*sp.cos(θ)))

    @staticmethod
    def kepler_equation(M, E, e):
        """
        Kepler's equation relating mean anomaly M and eccentric anomaly E:
            M = E - e sin E
        """
        return sp.Eq(M, E - e*sp.sin(E))


    @staticmethod
    def lagrangian(r_vec, v_vec, μ):
        """
        Lagrangian for a central force orbital problem:
            L = T - U
              = ½ |v|^2 + μ / |r|
        """
        v_squared = (v_vec.T * v_vec)[0,0]
        r_mag = sp.sqrt((r_vec.T * r_vec)[0,0])
        L = (1/2) * v_squared + μ / r_mag
        return L

    @staticmethod
    def hamiltonian(r_vec, p_vec, μ):
        """
        Hamiltonian formulation (energy function):
            H = T + U
              = ½ |p|^2 - μ / |r|
        """
        p_squared = (p_vec.T * p_vec)[0,0]
        r_mag = sp.sqrt((r_vec.T * r_vec)[0,0])
        H = (1/2) * p_squared - μ / r_mag
        return H
    
    @staticmethod
    def stable_orbit_transfer_solution(orbit1, orbit2):
        return OrbitalTransfer.symbolic_transfer_spline(orbit1, orbit2)
    
    @staticmethod
    def symbolic_orbit(name_prefix=''):
        """
        Constructs a universal symbolic representation of an orbit.
        Now includes center 'c' as well to support transfer computations.
        """
        r_vec = sp.MatrixSymbol(f'r_{name_prefix}', 3, 1)
        v_vec = sp.MatrixSymbol(f'v_{name_prefix}', 3, 1)
        μ = sp.Symbol(f'μ_{name_prefix}', positive=True)
        a = sp.Symbol(f'a_{name_prefix}', positive=True)
        e = sp.Symbol(f'e_{name_prefix}', positive=True)
        ε = sp.Symbol(f'ε_{name_prefix}', real=True)
        h = sp.MatrixSymbol(f'h_{name_prefix}', 3, 1)
        c = sp.MatrixSymbol(f'c_{name_prefix}', 3, 1)  # add center

        return {
            'r': r_vec,
            'v': v_vec,
            'μ': μ,
            'a': a,
            'e': e,
            'ε': ε,
            'h': h,
            'c': c
        }

# -------------------------------------------------
# Orbital Transfers
# -------------------------------------------------
from .orbital_transfer import OrbitalTransfer

# -------------------------------------------------
# (Optional) Relativistic Corrections
# -------------------------------------------------

class RelativisticCorrection:
    """
    Provides optional symbolic relativistic formulations.
    """

    @staticmethod
    def lorentz_gamma(v, c):
        """
        Lorentz factor:
            γ = 1 / sqrt(1 - v^2/c^2)
        """
        return 1 / sp.sqrt(1 - (v**2)/(c**2))

    @staticmethod
    def relativistic_energy(m, v, c):
        """
        Total energy:
            E = γ m c^2
        """
        γ = RelativisticCorrection.lorentz_gamma(v, c)
        return γ * m * c**2


# ------------------------------
# Example standalone symbolic demonstration
# ------------------------------

def demo():
    print("\n=== Canonical Orbital Demo ===")
    print("Vis-viva equation:")
    print(Orbit.vis_viva(μ, a, r))

    print("\nSpecific orbital energy:")
    print(Orbit.specific_orbital_energy(μ, a))

    print("\nOrbital shape equation:")
    print(Orbit.orbital_radius_theta(a, e, θ))

    print("\nKepler's equation:")
    print(Orbit.kepler_equation(M, E, e))

    # Example symbolic transfer spline system
    print("\nSymbolic orbital transfer system:")
    orbit1 = {'a': sp.Symbol('a1'), 'e': sp.Symbol('e1'),
              'μ': sp.Symbol('μ1'), 'c': sp.Matrix([0,0,0])}
    orbit2 = {'a': sp.Symbol('a2'), 'e': sp.Symbol('e2'),
              'μ': sp.Symbol('μ2'), 'c': sp.Matrix([10,0,0])}

    transfer_system = OrbitalTransfer.symbolic_transfer_spline(orbit1, orbit2)
    for key, value in transfer_system.items():
        print(f"\n-- {key} --")
        print(sp.pretty(value, use_unicode=True))

if __name__ == "__main__":
    demo()
