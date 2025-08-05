#!/usr/bin/env python3
"""
==========================================================
| DEC.py                                                 |
| Canonical symbolic Discrete Exterior Calculus (DEC)    |
| ----------------------------------------------------- |
| Pure symbolic textbook operators, implemented in sympy |
| for rigorous differential geometry on discrete spaces. |
==========================================================

References:
- Hirani, "Discrete Exterior Calculus"
- Desbrun et al., "Discrete Differential Forms for Computational Modeling"
- Abraham, Marsden, Ratiu, "Manifolds, Tensor Analysis, and Applications"
- Marsden & Hughes, "Mathematical Foundations of Elasticity"
"""

import sympy as sp

# ------------------------------------------------------
# DEC symbolic fundamental forms & standard symbols
# ------------------------------------------------------

class DEC:
    """
    Canonical symbolic Discrete Exterior Calculus operators.
    Provides symbolic definitions for:
      - 0-forms (scalar fields at vertices)
      - 1-forms (integrals along edges)
      - 2-forms (fluxes through faces)
      - Exterior derivative d
      - Hodge star ⋆
      - Codifferential δ
      - Laplace-de Rham Δ
    """
    # Standard symbolic indices for discrete geometry
    i, j, k = sp.symbols('i j k', integer=True)
    dx, dy, dz = sp.symbols('dx dy dz', real=True)
    ε = sp.Symbol('ε', real=True)  # small parameter for expansions

    # 0-form: scalar field on vertices
    @staticmethod
    def zero_form():
        φ = sp.Function('φ')
        return φ(DEC.i)

    # 1-form: integrals over edges
    @staticmethod
    def one_form():
        ω = sp.Function('ω')
        return ω(DEC.i, DEC.j)

    # 2-form: integrals over faces
    @staticmethod
    def two_form():
        σ = sp.Function('σ')
        return σ(DEC.i, DEC.j, DEC.k)

    # Exterior derivative d
    @staticmethod
    def exterior_derivative_0_to_1(φ):
        return φ(DEC.j) - φ(DEC.i)

    @staticmethod
    def exterior_derivative_1_to_2(ω):
        return ω(DEC.j, DEC.k) - ω(DEC.i, DEC.k) + ω(DEC.i, DEC.j)

    # Hodge star ⋆
    @staticmethod
    def hodge_star_0_to_n(φ, vol_dual_cell):
        return φ(DEC.i) * vol_dual_cell(DEC.i)

    @staticmethod
    def hodge_star_1_to_n_minus_1(ω, vol_dual_edge):
        return ω(DEC.i, DEC.j) * vol_dual_edge(DEC.i, DEC.j)

    # Codifferential δ
    @staticmethod
    def codifferential(star_ω_expr):
        return star_ω_expr - star_ω_expr.subs({DEC.i: DEC.j, DEC.j: DEC.i})

    # Laplace-de Rham Δ
    @staticmethod
    def laplace_de_rham(φ, vol_dual_cell, vol_dual_edge):
        dφ = DEC.exterior_derivative_0_to_1(φ)
        # Need to lambdify or inline dφ for hodge_star
        star_dφ = DEC.hodge_star_1_to_n_minus_1(
            lambda i, j: dφ.subs({DEC.i: i, DEC.j: j}), vol_dual_edge
        )
        δdφ = DEC.codifferential(star_dφ)
        return δdφ

    # Integration operators
    @staticmethod
    def integrate_over_vertex(φ, volume_weight):
        return φ(DEC.i) * volume_weight(DEC.i)

    @staticmethod
    def integrate_over_edge(ω, length_weight):
        return ω(DEC.i, DEC.j) * length_weight(DEC.i, DEC.j)

    @staticmethod
    def integrate_over_face(σ, area_weight):
        return σ(DEC.i, DEC.j, DEC.k) * area_weight(DEC.i, DEC.j, DEC.k)

    # --- Mesh-based symbolic dual constructions ---

    @staticmethod
    def circumcenter2D(i, j, k, V):
        """
        Symbolic circumcenter of triangle (i,j,k), where V is an IndexedBase (N,2)
        """
        x1, y1 = V[i, 0], V[i, 1]
        x2, y2 = V[j, 0], V[j, 1]
        x3, y3 = V[k, 0], V[k, 1]
        D = 2 * sp.det(sp.Matrix([
            [x1, y1, 1],
            [x2, y2, 1],
            [x3, y3, 1]
        ]))
        Ux = (1/D) * sp.det(sp.Matrix([
            [x1**2 + y1**2, y1, 1],
            [x2**2 + y2**2, y2, 1],
            [x3**2 + y3**2, y3, 1]
        ]))
        Uy = (1/D) * sp.det(sp.Matrix([
            [x1**2 + y1**2, x1, 1],
            [x2**2 + y2**2, x2, 1],
            [x3**2 + y3**2, x3, 1]
        ]))
        return sp.Matrix([Ux, Uy])

    @staticmethod
    def dual_edge_length(i, j, kL, kR, V):
        """
        Symbolic Voronoi dual edge length for primal edge (i, j),
        with left and right triangles (i, j, kL), (i, j, kR).
        """
        C_L = DEC.circumcenter2D(i, j, kL, V)
        C_R = DEC.circumcenter2D(i, j, kR, V)
        return sp.sqrt((C_L[0] - C_R[0])**2 + (C_L[1] - C_R[1])**2)

    @staticmethod
    def voronoi_cell_area(i, incident_tris, V):
        """
        Symbolic Voronoi dual cell area for vertex i,
        given a cyclic list of triangles around i, as (i, j, k).
        """
        area = 0
        circumcenters = []
        for tri in incident_tris:
            circumcenters.append(DEC.circumcenter2D(*tri, V))
        circumcenters.append(circumcenters[0])  # Close the loop
        for k in range(len(incident_tris)):
            C_k = circumcenters[k]
            C_k1 = circumcenters[k+1]
            x0, y0 = V[i, 0], V[i, 1]
            x1, y1 = C_k[0], C_k[1]
            x2, y2 = C_k1[0], C_k1[1]
            area += (x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1))/2
        return sp.simplify(area)

# ------------------------------------------------------
# Example symbolic DEC demonstration: continuous & mesh
# ------------------------------------------------------

def demo():
    print("\n=== Canonical DEC Symbolic Demo ===")

    φ = sp.Function('φ')
    ω = sp.Function('ω')
    σ = sp.Function('σ')
    vol_dual_cell = sp.Function('vol_dual_cell')
    vol_dual_edge = sp.Function('vol_dual_edge')

    print("\nExterior derivative dφ from 0-form to 1-form:")
    print(sp.pretty(DEC.exterior_derivative_0_to_1(φ), use_unicode=True))

    print("\nExterior derivative dω from 1-form to 2-form:")
    print(sp.pretty(DEC.exterior_derivative_1_to_2(ω), use_unicode=True))

    print("\nHodge star on 0-form to n-form dual:")
    print(sp.pretty(DEC.hodge_star_0_to_n(φ, vol_dual_cell), use_unicode=True))

    print("\nLaplace-de Rham Δφ (symbolic, general case):")
    Δφ = DEC.laplace_de_rham(φ, vol_dual_cell, vol_dual_edge)
    print(sp.pretty(Δφ, use_unicode=True))

    # === Mesh-based symbolic DEC: advanced expressions ===
    print("\n--- Mesh-based symbolic geometry (2D triangle mesh) ---")
    V = sp.IndexedBase('V')  # vertex positions V[i,0], V[i,1]

    # Example: Circumcenter of triangle (0,1,2)
    C = DEC.circumcenter2D(0,1,2,V)
    print("\nCircumcenter of triangle (0,1,2):")
    print(sp.pretty(C, use_unicode=True))

    # Dual edge length for edge (0,2) with triangles (0,2,1) and (0,2,3)
    dual_len = DEC.dual_edge_length(0,2,1,3,V)
    print("\nDual edge length for primal edge (0,2):")
    print(sp.pretty(dual_len, use_unicode=True))

    # Voronoi dual cell area at vertex 0, surrounded by tris (0,1,2), (0,2,3), (0,3,1)
    vor_area = DEC.voronoi_cell_area(0, [(0,1,2),(0,2,3),(0,3,1)], V)
    print("\nVoronoi dual cell area at vertex 0:")
    print(sp.pretty(vor_area, use_unicode=True))

    # Composition: Laplace-de Rham using *symbolic* mesh duals
    # (For devils: use DEC.laplace_de_rham, but supply custom vol_dual_edge from mesh as lambda)
    print("\nLaplace-de Rham on mesh: Δφ using mesh-derived dual edge lengths")
    mesh_vol_dual_edge = lambda i,j: DEC.dual_edge_length(i,j,1,3,V)  # For example
    Δφ_mesh = DEC.laplace_de_rham(φ, vol_dual_cell, mesh_vol_dual_edge)
    print(sp.pretty(Δφ_mesh, use_unicode=True))

if __name__ == "__main__":
    demo()
