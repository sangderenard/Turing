"""
Filtered Poisson on a Karate-Club-like graph (human-oriented demo)

- Uses your existing solver: src.common.tensors.filtered_poisson.filtered_poisson
- Uses AbstractTensor so everything stays in-backend
- Graph domain (mode="graph"): builds adjacency (A), constructs RHS, runs solver
- Prints residual diagnostics and a few interpretable stats
- Sign convention check: verifies whether L is "engineering" (D-A) or "mathematical" (-Δ)

Requirements:
- networkx (optional). If not available, a synthetic 2-community graph is used.
"""

from typing import Optional, Tuple

# Your code
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.filtered_poisson import filtered_poisson

# If you have a graph Laplacian helper class, you can optionally import it too.
# (Not required for this demo, but useful if you want to cross-check L explicitly.)
# from src.common.tensors.some_module import BuildGraphLaplace  # <- optional if present

# Third-party (optional)
try:
    import networkx as nx
    _HAVE_NX = True
except Exception:
    _HAVE_NX = False


def _karate_or_synthetic(n: int = 34, seed: int = 7) -> Tuple["AbstractTensor", str]:
    """
    Returns (A, label) where A is an AbstractTensor adjacency of shape (V,V).
    Prefers Zachary's Karate Club if networkx is installed; else builds a simple 2-community graph.
    """
    AT = AbstractTensor
    if _HAVE_NX:
        G = nx.karate_club_graph()
        nodes = sorted(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        V = len(nodes)
        A_np = [[0.0] * V for _ in range(V)]
        for u, v in G.edges():
            i, j = idx[u], idx[v]
            A_np[i][j] = 1.0
            A_np[j][i] = 1.0
        A = AT.tensor(A_np, dtype=AT.float_dtype_)
        return A, "Zachary's Karate Club (NetworkX)"
    else:
        # tiny synthetic 2-community random graph (no internet; no networkx)
        rng = 1234567 + seed
        AT.manual_seed(rng) if hasattr(AT, "manual_seed") else None
        V = n
        A = AT.zeros((V, V), dtype=AT.float_dtype_)
        # two groups
        half = V // 2
        # intra prob and inter prob (very light)
        p_intra = 0.22
        p_inter = 0.03
        # crude RNG using AbstractTensor if available, else Python's
        import random
        random.seed(seed)
        for i in range(V):
            for j in range(i + 1, V):
                same = (i < half and j < half) or (i >= half and j >= half)
                p = p_intra if same else p_inter
                if random.random() < p:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        # ensure diagonal is zero
        for i in range(V):
            A[i, i] = 0.0
        return A, "Synthetic 2-community (no networkx)"


def _laplacian_matvec(A: "AbstractTensor", x: "AbstractTensor") -> "AbstractTensor":
    """
    Compute y = L x with L = D - A (combinatorial graph Laplacian) using AbstractTensor ops only.
    This is *only for diagnostics*; the solver internally builds its own operator.
    """
    deg = A.sum(dim=-1)                    # (V,)
    Ax = A @ x                             # (V,) or (V,F)
    # broadcast deg over trailing dims if needed
    y = deg * x - Ax
    return y


def run_demo(
    filter_strength: float = 0.0,
    max_steps: int = 400,
    src_idx: Optional[int] = 0,
    sink_idx: Optional[int] = None,
) -> None:
    """
    Human-oriented runthrough:
      1) Build graph A
      2) Build RHS with +1 at src and -1 at sink
      3) Solve with filtered_poisson in graph mode (your code)
      4) Print residual diagnostics and sign-convention interpretation
    """
    AT = AbstractTensor

    # 1) Adjacency
    A, label = _karate_or_synthetic()
    V = A.shape[0] if hasattr(A, "shape") else A.shape[0]
    print(f"\n=== Filtered Poisson Demo (Graph Mode) ===")
    print(f"Graph: {label}  |  V={V}  |  E≈{int((A.sum().item() if hasattr(A.sum(), 'item') else float(A.sum()))/2)}")

    # 2) RHS: +1 at src, -1 at sink  (balanced → pure Poisson is solvable)
    if sink_idx is None:
        sink_idx = V - 1
    rhs = AT.zeros((V,), dtype=A.dtype)
    rhs[src_idx] = +1.0
    rhs[sink_idx] = -1.0

    # Report basic RHS stats
    rhs_sum = rhs.sum()
    rhs_sum_val = rhs_sum.item() if hasattr(rhs_sum, "item") else float(rhs_sum)
    print(f"RHS sum = {rhs_sum_val:+.3e}  (≈0 is ideal for unscreened Poisson)")

    # 3) Solve with your solver in *graph* mode.
    #    NOTE: We pass *only* what the library already supports.
    #    Your filtered_poisson should accept mode="graph" and 'adjacency=A'.
    #    We do not invent any functions here.
    u = filtered_poisson(
        rhs=rhs,
        mode="graph",
        adjacency=A,
        iterations=max_steps,
        filter_strength=filter_strength,
        tol=1e-10,
    )

    # If your filtered_poisson returns (u, stats), handle both shapes:
    if isinstance(u, tuple) and len(u) >= 1:
        u, stats = u[0], u[1]
    else:
        stats = None

    # 4) Diagnostics (purely in-demo; uses no new core functions)
    #    Build residual with the *engineering* Laplacian L = D - A
    Lu = _laplacian_matvec(A, u)
    r = rhs - Lu
    # norms
    r_abs = AT.abs(r) if hasattr(AT, "abs") else AT.tensor([abs(float(x)) for x in r])
    r_inf = r_abs.max()
    r2 = (r * r).sum()
    r_inf_val = r_inf.item() if hasattr(r_inf, "item") else float(r_inf)
    r2_val = (r2.sqrt().item() if hasattr(r2, "sqrt") else float(r2) ** 0.5)
    u_mean = (u.mean().item() if hasattr(u.mean(), "item") else float(u.mean()))
    u_min = (u.min().item() if hasattr(u.min(), "item") else float(u.min()))
    u_max = (u.max().item() if hasattr(u.max(), "item") else float(u.max()))

    print("\n--- Solution quality (interpreting L as D-A) ---")
    print(f"||L u - rhs||_inf = {r_inf_val:.3e}")
    print(f"||L u - rhs||_2   = {r2_val:.3e}")
    print(f"u: mean={u_mean:+.4f}  min={u_min:+.4f}  max={u_max:+.4f}")

    # Sign-convention note:
    # If your internal filtered_poisson solved (-Δ) u = rhs (i.e., L = D-A),
    # then the above residual will be near zero (good).
    # If your internal solver chose the opposite sign, the residual vs (-L) will be near zero instead.
    # We can also show that by flipping the sign in diagnostics:

    Lu_neg = -Lu
    r_neg = rhs - Lu_neg
    r_neg_inf = (AT.abs(r_neg).max().item()
                 if hasattr(AT.abs(r_neg).max(), "item")
                 else float(AT.abs(r_neg).max()))
    print("\n--- Sign-convention probe ---")
    print(f"With (-L): ||(-L) u - rhs||_inf = {r_neg_inf:.3e}")
    print("Interpretation:")
    if r_inf_val <= r_neg_inf:
        print("  Solver/graph convention consistent with L = D - A  (engineering / SPD).")
    else:
        print("  Solver/graph convention consistent with L = A - D  (mathematical ∇^2).")

    # Optional: a very small sanity check that x^T L x >= 0 for SPD L (=D-A)
    # (only as a quick energy diagnostic)
    x = rhs  # any nonzero vector is fine; reuse rhs
    Lx = Lu  # we already computed L u; reuse shape logic with x instead if preferred
    Lx = _laplacian_matvec(A, x)
    xtLx = (x * Lx).sum()
    xtLx_val = xtLx.item() if hasattr(xtLx, "item") else float(xtLx)
    print(f"\nEnergy check (x^T L x with L=D-A, x=rhs): {xtLx_val:+.6e} (should be ≥ 0 for SPD)")

    # Friendly pointers
    print("\nDone. Tips:")
    print("- Change src_idx/sink_idx to move sources/sinks and see value distribution shift.")
    print("- Increase max_steps or add regularization if available for faster convergence on open domains.")
    print("- If you add a Laplacian sign toggle in builders, this demo’s sign probe should flip accordingly.")


if __name__ == "__main__":
    # Defaults are conservative; tweak as desired.
    run_demo(
        filter_strength=0.0,
        max_steps=400,
        src_idx=0,
        sink_idx=None,  # default -> last node
    )
