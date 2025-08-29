"""
ndpca3_laplace_demo.py
----------------------

Demonstration using the proper pipeline:
- Build a PCA-based transform on a (U,V,W) grid
- Use geometry_factory.build_geometry to produce the metric-aware package
- Train a standalone NDPCA3Conv3d to fit a simple target defined on the grid

This verifies the NDPCA3Conv3d layer converges while receiving geometry from
the transform/Laplace package. No ad-hoc metric or shift logic outside the
layer — we follow the same structure as the Riemann demo, tailored for NDPCA3.
"""

from .ndpca3conv import NDPCA3Conv3d
from .ndpca3transform import PCABasisND, fit_metric_pca
from ..abstraction import AbstractTensor
from ..autograd import autograd
from ..riemann.geometry_factory import build_geometry
from ..abstract_nn.optimizer import Adam


def build_config():
    AT = AbstractTensor
    Nu = Nv = Nw = 8
    n = 8
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
        feats = [U, V, W, (U * V), (V * W), (W * U), (U.sin()), (V.cos())]
        return AT.stack(feats, dim=-1)

    config = {
        "geometry": {
            "key": "pca_nd",
            "grid_shape": (Nu, Nv, Nw),
            "boundary_conditions": (True,) * 6,
            "transform_args": {"pca_basis": basis, "phi_fn": phi_fn, "d_visible": 3},
            "laplace_kwargs": {},
        },
        "training": {
            "B": 4,
            "C": 2,
            "boundary_conditions": ("dirichlet",) * 6,
            "k": 3,
            "eig_from": "g",
            "pointwise": True,
        },
    }
    return config


def main(config=None):
    if config is None:
        config = build_config()
    geom_cfg = config["geometry"]
    train_cfg = config["training"]
    transform, grid_domain, package = build_geometry(geom_cfg)
    Nu, Nv, Nw = geom_cfg["grid_shape"]

    # Build teacher/student NDPCA3Conv3d and a supervised target on the grid
    B, C = train_cfg["B"], train_cfg["C"]
    like = AbstractTensor.get_tensor()
    layer = NDPCA3Conv3d(
        in_channels=C,
        out_channels=C,
        like=like,
        grid_shape=(Nu, Nv, Nw),
        boundary_conditions=train_cfg.get("boundary_conditions", ("dirichlet",) * 6),
        k=train_cfg.get("k", 3),
        eig_from=train_cfg.get("eig_from", "g"),
        pointwise=train_cfg.get("pointwise", True),
    )

    teacher = NDPCA3Conv3d(
        in_channels=C,
        out_channels=C,
        like=like,
        grid_shape=(Nu, Nv, Nw),
        boundary_conditions=train_cfg.get("boundary_conditions", ("dirichlet",) * 6),
        k=train_cfg.get("k", 3),
        eig_from=train_cfg.get("eig_from", "g"),
        pointwise=train_cfg.get("pointwise", True),
    )
    t_minus, t_center, t_plus = -0.6, 1.8, 0.7
    per_dir = [[t_minus / teacher.k, t_center / teacher.k, t_plus / teacher.k] for _ in range(teacher.k)]
    teacher.taps = AbstractTensor.tensor_from_list(per_dir, tape=autograd.tape, like=like, requires_grad=False)

    X = AbstractTensor.randn((B, C, Nu, Nv, Nw), requires_grad=True)
    with AbstractTensor.autograd.no_grad():
        target = teacher.forward(X, package=package)

    params = list(layer.parameters())
    opt = Adam(params, lr=1e-2)
    mse = lambda a, b: ((a - b) ** 2).mean()

    for epoch in range(1, 2001):
        for p in params:
            if hasattr(p, "zero_grad"):
                p.zero_grad()
        Y = layer.forward(X, package=package)
        loss = mse(Y, target)
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

