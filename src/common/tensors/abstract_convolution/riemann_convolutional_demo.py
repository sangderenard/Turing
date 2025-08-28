"""
riemann_convolutional_demo.py
----------------------------

Demo: Trains a RiemannConvolutional3D layer on a synthetic regression task, ensuring that:
- The LocalStateNetwork parameters (in the Laplacian) are updated during training.
- The metric at each location factors into the convolution kernel.
- The geometry is driven by the transform + Laplacian pipeline.
- Training proceeds until loss < 1e-6 (or max epochs).

This demo audits the full geometry-driven learning pipeline.
"""


from .riemann_convolutional import RiemannConvolutional3D
from .ndpca3transform import PCABasisND, fit_metric_pca, PCANDTransform
from ..abstraction import AbstractTensor
from src.common.tensors.abstract_nn.optimizer import Adam
import numpy as np

# --- 1. Build a synthetic PCA transform and grid ---
def build_transform_and_grid(Nu=8, Nv=8, Nw=8, n=8):
    AT = AbstractTensor
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
    return xform, (Nu, Nv, Nw)

# --- 2. Demo training loop ---
def main():
    AT = AbstractTensor
    xform, grid_shape = build_transform_and_grid(Nu=8, Nv=8, Nw=8)
    B, C = 4, 3
    layer = RiemannConvolutional3D(
        in_channels=C,
        out_channels=C,
        grid_shape=grid_shape,
        transform=xform,
        boundary_conditions=("dirichlet",)*6,
        k=3,
        eig_from="g",
        pointwise=True,
    )
    # Target: simple function of grid (e.g., sum of coordinates)
    U, V, W = layer.grid_domain.U, layer.grid_domain.V, layer.grid_domain.W
    target = (U + V + W).unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
    # Input: random
    x = AT.randn((B, C, *grid_shape), requires_grad=True)
    # --- Parameter and gradient collection helpers ---
    from ..logger import get_tensors_logger
    logger = get_tensors_logger()
    def collect_params_and_grads():
        params, grads = [], []
        # Convolutional weights
        if hasattr(layer.conv, 'parameters'):
            for p in layer.conv.parameters():
                params.append(p)
                grads.append(getattr(p, 'grad', None))
        # LocalStateNetwork (if present)
        lsn = layer.laplace_package.get('local_state_network', None) if isinstance(layer.laplace_package, dict) else None
        if lsn and hasattr(lsn, 'parameters'):
            for p in lsn.parameters():
                params.append(p)
                grads.append(getattr(p, 'grad', None))
        # Fallback: any other objects with .parameters
        if isinstance(layer.laplace_package, dict):
            for v in layer.laplace_package.values():
                if hasattr(v, 'parameters'):
                    for p in v.parameters():
                        params.append(p)
                        grads.append(getattr(p, 'grad', None))
        # Log all params and grads, including Nones
        for i, (p, g) in enumerate(zip(params, grads)):
            label = getattr(p, '_label', None)
            logger.info(f"Param {i}: label={label}, shape={getattr(p, 'shape', None)}, grad is None={g is None}, grad shape={getattr(g, 'shape', None) if g is not None else None}")
        return params, grads

    params, _ = collect_params_and_grads()
    optimizer = Adam(params, lr=1e-3)
    loss_fn = lambda y, t: ((y - t) ** 2).mean()
    for epoch in range(1, 10001):
        # Zero gradients for all params
        for p in params:
            if hasattr(p, 'zero_grad'):
                p.zero_grad()
            elif hasattr(p, 'grad'):
                p.grad = AbstractTensor.zeros_like(p.grad)
        y = layer.forward(x)
        loss = loss_fn(y, target)
        # Backward pass (assume .backward() populates .grad)
        if hasattr(loss, 'backward'):
            loss.backward()
        # Re-collect params and grads (in case new tensors were created)
        params, grads = collect_params_and_grads()
        for p in params:
            label = getattr(p, '_label', None)
            assert hasattr(p, 'grad'), f"Parameter {label or p} has no grad attribute"
            assert p.grad is not None, f"Parameter {label or p} grad is None after backward()"
            assert p.grad.shape == p.shape, f"Parameter {label or p} has incorrect grad shape: grad.shape={getattr(p.grad, 'shape', None)}, param.shape={getattr(p, 'shape', None)}"
        for g, p in zip(grads, params):
            label = getattr(p, '_label', None)
            assert hasattr(g, 'shape'), f"Gradient for {label or p} has no shape attribute"
            assert g.shape == p.shape, f"Gradient for {label or p} has incorrect shape: grad.shape={getattr(g, 'shape', None)}, param.shape={getattr(p, 'shape', None)}"
        optimizer.step(params, grads)
        if epoch % 100 == 0 or loss.item() < 1e-6:
            print(f"Epoch {epoch}: loss={loss.item():.2e}")
        if loss.item() < 1e-6:
            print("Converged.")
            break
    # Audit: check that metric and local state network are used
    print("Metric at center voxel:", layer.laplace_package['metric']['g'][grid_shape[0]//2, grid_shape[1]//2, grid_shape[2]//2])
    if 'local_state_network' in layer.laplace_package:
        print("LocalStateNetwork parameters:", list(layer.laplace_package['local_state_network'].parameters()))

if __name__ == "__main__":
    main()
