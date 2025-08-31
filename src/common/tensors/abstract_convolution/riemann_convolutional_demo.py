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


from .metric_steered_conv3d import MetricSteeredConv3DWrapper
from .ndpca3transform import PCABasisND, fit_metric_pca
from ..abstraction import AbstractTensor
from ..autograd import autograd
from ..riemann.geometry_factory import build_geometry
from src.common.tensors.abstract_nn.optimizer import Adam
import numpy as np


def build_config():
    AT = AbstractTensor
    Nu = Nv = Nw = 8
    n = 8
    B = 50000
    t = AT.arange(0, B, 1, requires_grad=True)
    autograd.tape.annotate(t, label="riemann_demo.t_arange")
    autograd.tape.auto_annotate_eval(t)
    t = (t / (B - 1) - 0.5) * 6.283185307179586
    autograd.tape.annotate(t, label="riemann_demo.t_scaled")
    autograd.tape.auto_annotate_eval(t)
    base = AT.stack([
        t.sin(), t.cos(), (2 * t).sin(), (0.5 * t).cos(),
        (0.3 * t).sin(), (1.7 * t).cos(), (0.9 * t).sin(), (1.3 * t).cos()
    ], dim=-1)
    autograd.tape.annotate(base, label="riemann_demo.base")
    autograd.tape.auto_annotate_eval(base)
    scale = AT.get_tensor([2.0, 1.5, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1], requires_grad=True)
    autograd.tape.annotate(scale, label="riemann_demo.scale")
    autograd.tape.auto_annotate_eval(scale)
    u_samples = base * scale
    autograd.tape.annotate(u_samples, label="riemann_demo.u_samples")
    autograd.tape.auto_annotate_eval(u_samples)
    weights = (-(t**2)).exp()
    autograd.tape.annotate(weights, label="riemann_demo.weights")
    autograd.tape.auto_annotate_eval(weights)
    M = AT.eye(n)
    autograd.tape.annotate(M, label="riemann_demo.M_eye")
    autograd.tape.auto_annotate_eval(M)
    diag = AT.get_tensor([1.0, 0.5, 0.25, 2.0, 1.0, 3.0, 0.8, 1.2], requires_grad=True)
    autograd.tape.annotate(diag, label="riemann_demo.diag")
    autograd.tape.auto_annotate_eval(diag)
    M = M * diag.reshape(1, -1)
    M = M.swapaxes(-1, -2) * diag.reshape(1, -1)
    autograd.tape.annotate(M, label="riemann_demo.metric_M")
    autograd.tape.auto_annotate_eval(M)
    basis = fit_metric_pca(u_samples, weights=weights, metric_M=M)
    autograd.tape.annotate(basis, label="riemann_demo.basis")
    autograd.tape.auto_annotate_eval(basis)

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
            "C": 3,
            "boundary_conditions": ("dirichlet",) * 6,
            "k": 3,
            "eig_from": "g",
            "pointwise": True,
        },
    }
    return config


def main(config=None):
    AT = AbstractTensor
    if config is None:
        config = build_config()
    geom_cfg = config["geometry"]
    train_cfg = config["training"]
    transform, grid, _ = build_geometry(geom_cfg)
    grid_shape = geom_cfg["grid_shape"]
    B, C = train_cfg["B"], train_cfg["C"]
    layer = MetricSteeredConv3DWrapper(
        in_channels=C,
        out_channels=C,
        grid_shape=grid_shape,
        transform=transform,
        boundary_conditions=train_cfg.get("boundary_conditions", ("dirichlet",) * 6),
        k=train_cfg.get("k", 3),
        eig_from=train_cfg.get("eig_from", "g"),
        pointwise=train_cfg.get("pointwise", True),
        deploy_mode="modulated",
        laplace_kwargs={"lambda_reg": 0.5},
    )
    from ..abstraction import AbstractTensor as _AT
    grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
    print(f"[DEBUG] LSN instance id at layer creation: {id(layer.local_state_network)} | grad_tracking_enabled={grad_enabled}")
    U, V, W = grid.U, grid.V, grid.W
    autograd.tape.annotate(U, label="riemann_demo.grid_U")
    autograd.tape.auto_annotate_eval(U)
    autograd.tape.annotate(V, label="riemann_demo.grid_V")
    autograd.tape.auto_annotate_eval(V)
    autograd.tape.annotate(W, label="riemann_demo.grid_W")
    autograd.tape.auto_annotate_eval(W)
    target = (U + V + W).unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
    autograd.tape.annotate(target, label="riemann_demo.target")
    autograd.tape.auto_annotate_eval(target)
    x = AT.randn((B, C, *grid_shape), requires_grad=True)
    autograd.tape.annotate(x, label="riemann_demo.input")
    autograd.tape.auto_annotate_eval(x)
    # When AUTOGRAD_STRICT=1, unused tensors trigger connectivity errors.
    # Uncomment one of the lines below to relax those checks:
    # autograd.strict = False                   # disable strict mode globally
    # autograd.whitelist(x, target)             # or whitelist specific tensors
    # autograd.whitelist_labels(r"riemann_demo.*")  # whitelist by label pattern
    # --- Parameter and gradient collection helpers ---
    from ..logger import get_tensors_logger
    logger = get_tensors_logger()
    def collect_params_and_grads():
        params, grads = [], []
        # Convolutional weights
        if hasattr(layer.conv, 'parameters'):
            for p in layer.conv.parameters():
                params.append(p)
                grads.append(getattr(p, '_grad', None))
        # LocalStateNetwork (if present)
        lsn = layer.local_state_network if hasattr(layer, 'local_state_network') else None
        if lsn is None:
            raise ValueError("LocalStateNetwork not found")
        if lsn and hasattr(lsn, 'parameters'):
            for p in lsn.parameters(include_all=True):
                params.append(p)
                grads.append(getattr(p, '_grad', None))
        else:
            raise ValueError("LocalStateNetwork not found")
        # Fallback: any other objects with .parameters
        if isinstance(layer.laplace_package, dict):
            for v in layer.laplace_package.values():
                if hasattr(v, 'parameters'):
                    for p in v.parameters():
                        params.append(p)
                        grads.append(getattr(p, '_grad', None))
        # Log all params and grads, including Nones
        for i, (p, g) in enumerate(zip(params, grads)):
            label = getattr(p, '_label', None)
            logger.info(f"Param {i}: label={label}, shape={getattr(p, 'shape', None)}, grad is None={g is None}, grad shape={getattr(g, 'shape', None) if g is not None else None}")
        return params, grads
    y = layer.forward(x)
    grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
    print(f"[DEBUG] LSN instance id after forward: {id(layer.local_state_network)} | grad_tracking_enabled={grad_enabled}")
    print(f"[DEBUG] LSN param ids: {[id(p) for p in layer.local_state_network.parameters(include_all=True)]}")
    print(f"[DEBUG] LSN param requires_grad: {[getattr(p, 'requires_grad', None) for p in layer.local_state_network.parameters(include_all=True)]}")
    print(f"[DEBUG] LSN _regularization_loss: {layer.local_state_network._regularization_loss}")
    print(f"[DEBUG] LSN _regularization_loss grad_fn: {getattr(layer.local_state_network._regularization_loss, 'grad_fn', None)}")
    grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
    print(f"[DEBUG] About to call backward on LSN _regularization_loss | grad_tracking_enabled={grad_enabled}")
    lsn = layer.local_state_network
    lsn._regularization_loss.backward()
    grad_w = getattr(lsn._weighted_padded, '_grad', AbstractTensor.zeros_like(lsn._weighted_padded))
    grad_m = getattr(lsn._modulated_padded, '_grad', AbstractTensor.zeros_like(lsn._modulated_padded))
    lsn.backward(grad_w, grad_m, lambda_reg=0.5)
    for i, p in enumerate(lsn.parameters(include_all=True)):
        grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
        print(
            f"[DEBUG] After backward: param {i} id={id(p)} grad={getattr(p, '_grad', None)} | grad_tracking_enabled={grad_enabled}"
        )

    params, _ = collect_params_and_grads()
    optimizer = Adam(params, lr=1e-2)
    loss_fn = lambda y, t: ((y - t) ** 2).mean()
    for epoch in range(1, 10001):
        # Zero gradients for all params
        for p in params:
            if hasattr(p, 'zero_grad'):
                p.zero_grad()
            elif hasattr(p, '_grad'):
                p._grad = AbstractTensor.zeros_like(p._grad)
        y = layer.forward(x)
        autograd.tape.auto_annotate_eval(y)
        loss = loss_fn(y, target)
        LSN_loss = layer.local_state_network._regularization_loss
        print(f"Epoch {epoch}: loss={loss.item():.2e}, LSN_loss={LSN_loss.item():.2e}")
        loss = LSN_loss + loss
        print(f"Total loss={loss.item():.2e}")
        autograd.tape.annotate(loss, label="riemann_demo.loss")
        autograd.tape.auto_annotate_eval(loss)
        # layer.report_orphan_nodes()  # retired / no-op
        # Backward pass (assume .backward() populates ._grad)
        loss.backward()
        lsn = layer.local_state_network
        grad_w = getattr(lsn._weighted_padded, '_grad', AbstractTensor.zeros_like(lsn._weighted_padded))
        grad_m = getattr(lsn._modulated_padded, '_grad', AbstractTensor.zeros_like(lsn._modulated_padded))
        lsn.backward(grad_w, grad_m, lambda_reg=0.5)
        # Re-collect params and grads (in case new tensors were created)
        params, grads = collect_params_and_grads()
        for p in params:
            label = getattr(p, '_label', None)
            # print(p)
            # print(p._grad)
            assert hasattr(p, '_grad'), f"Parameter {label or p} has no grad attribute"
            
            #assert p._grad is not None, f"Parameter {label or p} grad is None after backward()"
            #assert p._grad.shape == p.shape, f"Parameter {label or p} has incorrect grad shape: grad.shape={getattr(p._grad, 'shape', None)}, param.shape={getattr(p, 'shape', None)}"
        for i, (g, p) in enumerate(zip(grads, params)):
            if g is None:
                g = AbstractTensor.zeros_like(p)
                grads[i] = g
            label = getattr(p, '_label', None)
            #assert hasattr(g, 'shape'), f"Gradient for {label or p} has no shape attribute"
            #assert g.shape == p.shape, f"Gradient for {label or p} has incorrect shape: grad.shape={getattr(g, 'shape', None)}, param.shape={getattr(p, 'shape', None)}"
        # Optimizer returns updated tensors; copy values in-place to preserve
        # parameter identity on the tape so they remain registered.
        new_params = optimizer.step(params, grads)
        from ..abstraction import AbstractTensor as _AT
        for p, new_p in zip(params, new_params):
            _AT.copyto(p, new_p)
        if epoch % 1 == 0 or loss.item() < 1e-6:
            print(f"Epoch {epoch}: loss={loss.item():.2e}")
            # Gradient report
            for i, (p, g) in enumerate(zip(params, grads)):
                label = getattr(p, '_label', f'param_{i}')
                if g is not None:
                    try:
                        g_np = g.data if hasattr(g, 'data') else g
                        g_mean = g_np.mean() if hasattr(g_np, 'mean') else 'n/a'
                        g_norm = (g_np ** 2).sum() ** 0.5 if hasattr(g_np, '__pow__') else 'n/a'
                        print(f"  Grad {i} ({label}): mean={g_mean:.2e}, norm={g_norm:.2e}")
                    except Exception as e:
                        print(f"  Grad {i} ({label}): error reporting grad: {e}")
                else:
                    print(f"  Grad {i} ({label}): None")
        if loss.item() < 1e-6:
            print("Converged.")
            break
    # Audit: check that metric and local state network are used
    print("Metric at center voxel:", layer.laplace_package['metric']['g'][grid_shape[0]//2, grid_shape[1]//2, grid_shape[2]//2])
    if 'local_state_network' in layer.laplace_package:
        print("LocalStateNetwork parameters:", list(layer.laplace_package['local_state_network'].parameters()))

if __name__ == "__main__":
    main()
