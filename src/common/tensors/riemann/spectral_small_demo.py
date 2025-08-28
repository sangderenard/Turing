"""
Spectral Small-Grid Demo
------------------------

Builds a small (8^3) manifold via PCANDTransform + GridDomain + BuildLaplace3D,
computes a small set of LB eigenpairs, and trains a SpectralConv3D (student) to
match a fixed SpectralConv3D (teacher) on random inputs. Demonstrates correct
autograd wiring and convergence of the spectral mapping.
"""

from __future__ import annotations

from ..abstraction import AbstractTensor as AT
from ..autograd import autograd
from ..abstract_nn.optimizer import Adam
from ..abstract_convolution.laplace_nd import GridDomain
from ..abstract_convolution.ndpca3transform import fit_metric_pca, PCANDTransform
from .manifold import ManifoldPackage
from .spectral import SpectralConv3D


def build_transform_and_grid(Nu=8, Nv=8, Nw=8, n=8):
    # Synthesize intrinsic samples
    B = 500
    t = AT.arange(0, B, 1)
    t = (t / (B - 1) - 0.5) * 6.283185307179586
    base = AT.stack([
        t.sin(), t.cos(), (2 * t).sin(), (0.5 * t).cos(),
        (0.3 * t).sin(), (1.7 * t).cos(), (0.9 * t).sin(), (1.3 * t).cos()
    ], dim=-1)
    scale = AT.get_tensor([2.0, 1.5, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1])
    u_samples = base * scale
    weights = (-(t**2)).exp()
    # Metric M for metric-PCA fit (diagonal SPD)
    M = AT.eye(n)
    diag = AT.get_tensor([1.0, 0.5, 0.25, 2.0, 1.0, 3.0, 0.8, 1.2])
    M = M * diag.reshape(1, -1)
    M = M.swapaxes(-1, -2) * diag.reshape(1, -1)
    basis = fit_metric_pca(u_samples, weights=weights, metric_M=M)

    # Define visible embedding phi(U,V,W) in R^n
    def phi_fn(U, V, W):
        feats = [
            U, V, W, (U*V), (V*W), (W*U), (U.sin()), (V.cos())
        ]
        return AT.stack(feats, dim=-1)

    xform = PCANDTransform(basis, phi_fn, d_visible=3)

    # Build (U,V,W) grid domain
    U = AT.linspace(-1.0, 1.0, Nu).reshape(Nu, 1, 1) * AT.ones((1, Nv, Nw))
    V = AT.linspace(-1.0, 1.0, Nv).reshape(1, Nv, 1) * AT.ones((Nu, 1, Nw))
    W = AT.linspace(-1.0, 1.0, Nw).reshape(1, 1, Nw) * AT.ones((Nu, Nv, 1))
    grid = GridDomain(
        U, V, W,
        grid_boundaries=(True,) * 6,
        transform=xform,
        coordinate_system="rectangular",
    )
    return xform, grid


def main():
    Nu = Nv = Nw = 8
    Cin = 2
    Cout = 2
    K = 16  # number of LB modes
    B = 4

    # Transform + grid + manifold (with eigenpairs)
    xform, grid = build_transform_and_grid(Nu=Nu, Nv=Nv, Nw=Nw)
    manifold = ManifoldPackage(xform, grid, num_eigenpairs=K)
    manifold.build()

    # Teacher and student spectral convs
    teacher = SpectralConv3D(Cin, Cout, num_modes=K)
    student = SpectralConv3D(Cin, Cout, num_modes=K)

    # Inputs (fixed for training)
    X = AT.randn((B, Cin, Nu, Nv, Nw), requires_grad=True)

    # Teacher target
    with autograd.no_grad():
        Yt = teacher.forward(X, manifold=manifold)

    # Train student to match teacher
    params = list(student.parameters())
    opt = Adam(params, lr=1e-2)
    mse = lambda a, b: ((a - b) ** 2).mean()

    for epoch in range(1, 2001):
        for p in params:
            if hasattr(p, "zero_grad"):
                p.zero_grad()
        Yp = student.forward(X, manifold=manifold)
        loss = mse(Yp, Yt)
        autograd.grad(loss, params, retain_graph=False, allow_unused=False)
        new_params = opt.step(params, [p.grad for p in params])
        for p, np_ in zip(params, new_params):
            AT.copyto(p, np_)
        if epoch % 100 == 0 or float(loss.item()) < 1e-6:
            print(f"Epoch {epoch}: loss={float(loss.item()):.3e}")
        if float(loss.item()) < 1e-5:
            print("Converged.")
            break


if __name__ == "__main__":
    main()

