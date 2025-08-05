import sympy
from sympy import Sum, IndexedBase, Idx, symbols, Function
import numpy as np
from ..physics import (
    vis_viva_eq,
    sch_eq,
    efe,
    laplace_de_rham,
    fourier_transform,
    A, B,
    maxwell_eq,
    field_tensor,
    Tμν_em,
    wave_eq,
    dirac_eq,
)
from ..dec import DEC
from ..orbital import Orbit
from .graph_express_chalkboard_problem import chalkboard_problem

M_val, N_val, K_val = 3, 6, 9
M, N, K = symbols('M N K', integer=True)
i, j, k = Idx('i', M), Idx('j', N), Idx('k', K)
A_t, B_t, C_t = IndexedBase('A'), IndexedBase('B'), IndexedBase('C')
x, y, z = symbols('x y z', real=True)

test_suite = [
    {
        'name': "1 + 1",
        'expr_fn': lambda: sympy.Integer(1) + sympy.Integer(1),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: 2
    },
    {
        'name': "(x + y) * z (irreducible)",
        'expr_fn': lambda: (sympy.Symbol('x') + sympy.Symbol('y')) * sympy.Symbol('z'),
        'dims': (1,),
        'data_sources': lambda: {'x': 2, 'y': 3, 'z': 4},
        'expected_fn': lambda ds: (ds['x'] + ds['y']) * ds['z']
    },
    {
        'name': "Elementwise A + C",
        'expr_fn': lambda ii, jj: A_t[ii, jj] + C_t[ii, jj],
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'C': np.random.rand(M_val, N_val)
        },
        'expected_fn': lambda ds: ds['A'] + ds['C']
    },
    {
        'name': "Dot product across K",
        'expr_fn': lambda ii, jj: Sum(A_t[ii, k] * B_t[k, jj], (k, 0, K_val - 1)),
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, K_val),
            'B': np.random.rand(K_val, N_val)
        },
        'expected_fn': lambda ds: np.dot(ds['A'], ds['B'])
    },
    {
        'name': "Elementwise A * C",
        'expr_fn': lambda ii, jj: A_t[ii, jj] * C_t[ii, jj],
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'C': np.random.rand(M_val, N_val)
        },
        'expected_fn': lambda ds: ds['A'] * ds['C']
    },
    {
        'name': "(A + C) ** 2",
        'expr_fn': lambda ii, jj: (A_t[ii, jj] + C_t[ii, jj])**2,
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'C': np.random.rand(M_val, N_val)
        },
        'expected_fn': lambda ds: (ds['A'] + ds['C']) ** 2
    },
    {
        'name': "3D Tensor Outer Product",
        'expr_fn': lambda ii, jj, kk: A_t[ii, jj] * B_t[kk, jj] + C_t[ii, kk],
        'dims': (M_val, N_val, K_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'B': np.random.rand(K_val, N_val),
            'C': np.random.rand(M_val, K_val)
        },
        'expected_fn': lambda ds: np.einsum('ij,kj->ikj', ds['A'], ds['B']) + np.broadcast_to(ds['C'][:, None, :], (M_val, N_val, K_val))
    },
    {
        'name': "Elementwise Sin + Exp",
        'expr_fn': lambda ii, jj: sympy.sin(A_t[ii, jj]) + sympy.exp(C_t[ii, jj]),
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'C': np.random.rand(M_val, N_val)
        },
        'expected_fn': lambda ds: np.sin(ds['A']) + np.exp(ds['C'])
    },
    {
        'name': "DEC Laplace-de Rham (symbolic, generic)",
        'expr_fn': lambda: DEC.laplace_de_rham(Function('φ'), Function('vol_dual_cell'), Function('vol_dual_edge')),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Laplace-de Rham (mesh dual functions)",
        'expr_fn': lambda: DEC.laplace_de_rham(Function('φ'), DEC.mesh_voronoi_cell, DEC.mesh_dual_edge),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Voronoi Dual Cell Area (vertex 0, tris)",
        'expr_fn': lambda: DEC.mesh_voronoi_cell(0, [(0,1,2),(0,2,3),(0,3,1)], sympy.IndexedBase('V')),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Dual Edge Length (0,2,1,3)",
        'expr_fn': lambda: DEC.mesh_dual_edge(0, 2, 1, 3, sympy.IndexedBase('V')),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Circumcenter2D (0,1,2)",
        'expr_fn': lambda: DEC.circumcenter2D(0, 1, 2, sympy.IndexedBase('V')),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Symbolic Fourier Transform (from physics.py)",
        'expr_fn': lambda: fourier_transform,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Vis Viva Equation (Orbital Mechanics)",
        'expr_fn': lambda: vis_viva_eq,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Schrödinger Equation (physics.py)",
        'expr_fn': lambda: sch_eq,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Einstein Field Equation (symbolic, 4x4)",
        'expr_fn': lambda: efe,
        'dims': (4, 4),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Laplace-de Rham (physics.py symbol)",
        'expr_fn': lambda: laplace_de_rham,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Magnetic Vector Potential (A)",
        'expr_fn': lambda: A,
        'dims': (3,),  # 3-vector
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Magnetic Field (B = curl A)",
        'expr_fn': lambda: B,
        'dims': (3,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Maxwell Equation (covariant symbolic)",
        'expr_fn': lambda: maxwell_eq,
        'dims': (4, 4),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Field Tensor F^{μν}=∂^μA^ν-∂^νA^μ",
        'expr_fn': lambda: field_tensor,
        'dims': (4, 4),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Stress-Energy Tensor T^{μν}",
        'expr_fn': lambda: Tμν_em,
        'dims': (4, 4),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Wave Equation (physics.py)",
        'expr_fn': lambda: wave_eq,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Dirac Equation (physics.py)",
        'expr_fn': lambda: dirac_eq,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Orbital symbolic transfer EOM",
        'expr_fn': lambda: Orbit.stable_orbit_transfer_solution(
            Orbit.symbolic_orbit('1'),
            Orbit.symbolic_orbit('2')
        )['equation_of_motion'],
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Chalkboard Problem",
        'expr_fn': lambda: chalkboard_problem,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
]
