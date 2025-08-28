"""
ndpca3_laplace_demo.py
----------------------

Demonstration using the proper pipeline:
- Build a PCA-based transform on a (U,V,W) grid
- Use BuildLaplace3D to produce a metric-aware package (g, inv_g, etc.)
- Train a standalone NDPCA3Conv3d to fit a simple target defined on the grid

This verifies the NDPCA3Conv3d layer converges while receiving geometry from
the transform/Laplace package. No ad-hoc metric or shift logic outside the
layer — we follow the same structure as the Riemann demo, tailored for NDPCA3.
"""

from .laplace_nd import BuildLaplace3D, GridDomain
from .ndpca3conv import NDPCA3Conv3d
from .ndpca3transform import PCABasisND, fit_metric_pca, PCANDTransform
from ..abstraction import AbstractTensor
from ..autograd import autograd
from ..abstract_nn.optimizer import Adam


def build_pca_transform_and_grid(Nu=8, Nv=8, Nw=8, n=8):
    AT = AbstractTensor
    # Synthesize intrinsic samples (B, n)
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
    M = AT.eye(n)
    diag = AT.get_tensor([1.0, 0.5, 0.25, 2.0, 1.0, 3.0, 0.8, 1.2])
    M = M * diag.reshape(1, -1)
    M = M.swapaxes(-1, -2) * diag.reshape(1, -1)
    basis = fit_metric_pca(u_samples, weights=weights, metric_M=M)
    def phi_fn(U, V, W):
        feats = [U, V, W, (U*V), (V*W), (W*U), (U.sin()), (V.cos())]
        return AT.stack(feats, dim=-1)
    xform = PCANDTransform(basis, phi_fn, d_visible=3)
    # Now build the canonical grid using GridDomain
    grid_domain = GridDomain(
        AT.linspace(-1.0, 1.0, Nu).reshape(Nu, 1, 1) * AT.ones((1, Nv, Nw)),
        AT.linspace(-1.0, 1.0, Nv).reshape(1, Nv, 1) * AT.ones((Nu, 1, Nw)),
        AT.linspace(-1.0, 1.0, Nw).reshape(1, 1, Nw) * AT.ones((Nu, Nv, 1)),
        grid_boundaries=(True,)*6,
        transform=xform,
        coordinate_system="rectangular"
    )
    return xform, grid_domain


pass  # replaced by build_pca_transform_and_grid


def build_laplace_package(grid_domain, xform, boundary_conditions=("dirichlet",)*6):
    builder = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,
        precision=getattr(AbstractTensor, "float_dtype_", None) or grid_domain.U.dtype,
        resolution=grid_domain.U.shape[0],
        metric_tensor_func=xform.metric_tensor_func,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10,
        device=getattr(grid_domain.U, "device", None),
    )
    _, _, package = builder.build_general_laplace(grid_domain.U, grid_domain.V, grid_domain.W, return_package=True)
    return package


def main():
    # 1) Transform + grid + Laplace package
    Nu = Nv = Nw = 8
    xform, grid_domain = build_pca_transform_and_grid(Nu=Nu, Nv=Nv, Nw=Nw)
    package = build_laplace_package(grid_domain, xform)

    # 2) Build teacher/student NDPCA3Conv3d and a supervised target on the grid
    # Target must depend on X via the same operator family; otherwise learning stalls.
    B, C = 4, 2
    like = AbstractTensor.get_tensor()
    # Student to train
    layer = NDPCA3Conv3d(
        in_channels=C,
        out_channels=C,
        like=like,
        grid_shape=(grid_domain.resolution_u, grid_domain.resolution_v, grid_domain.resolution_w),
        boundary_conditions=("dirichlet",)*6,
        k=3,
        eig_from="g",
        pointwise=True,
    )

    # Teacher with fixed taps generates the target Y from X using the same package
    teacher = NDPCA3Conv3d(
        in_channels=C,
        out_channels=C,
        like=like,
        grid_shape=(grid_domain.resolution_u, grid_domain.resolution_v, grid_domain.resolution_w),
        boundary_conditions=("dirichlet",)*6,
        k=3,
        eig_from="g",
        pointwise=True,
    )
    # Set teacher taps so that column sums equal desired 3‑tap values
    t_minus, t_center, t_plus = -0.6, 1.8, 0.7
    per_dir = [[t_minus / teacher.k, t_center / teacher.k, t_plus / teacher.k] for _ in range(teacher.k)]
    teacher.taps = AbstractTensor.tensor_from_list(per_dir, tape=autograd.tape, like=like, requires_grad=False)
    
    # Inputs: random but fixed for training
    X = AbstractTensor.randn((B, C, Nu, Nv, Nw), requires_grad=True)

    # Target: teacher conv applied to X under the same geometry
    with AbstractTensor.autograd.no_grad():
        target = teacher.forward(X, package=package)

    # 3) Train taps (and optional pointwise) to fit target
    params = list(layer.parameters())
    opt = Adam(params, lr=1e-2)
    mse = lambda a, b: ((a - b) ** 2).mean()

    for epoch in range(1, 2001):
        # Zero gradients
        for p in params:
            if hasattr(p, "zero_grad"):
                p.zero_grad()
        Y = layer.forward(X, package=package)
        loss = mse(Y, target)
        # Compute grads explicitly for these params to avoid registry issues
        autograd.grad(loss, params, retain_graph=False, allow_unused=False)
        new_params = opt.step(params, [p.grad for p in params])
        for p, np_ in zip(params, new_params):
            AbstractTensor.copyto(p, np_)
        if epoch % 100 == 0 or float(loss.item()) < 1e-6:
            print(f"Epoch {epoch}: loss={float(loss.item()):.3e}")
        if float(loss.item()) < 1e-5:
            print("Converged.")
            break

if __name__ == "__main__":
    main()
