"""
Demo: MetricSteeredConv3DWrapper pipeline with real Laplace, real metric, and gradient check.

This script runs a forward and backward pass through the full metric-steered pipeline and prints a human-friendly summary of parameter gradients.
"""

from src.common.tensors.abstract_convolution.metric_steered_conv3d import MetricSteeredConv3DWrapper
from src.common.tensors.abstract_convolution.laplace_nd import RectangularTransform
from src.common.tensors.abstraction import AbstractTensor



def learnable_metric_tensor_func(u, v, w, dxdu, dydu, dzdu, dxdv, dydv, dzdv, dxdw, dydw, dzdw):
    # Simple learnable metric: g = a*I + b*outer(u,v,w)
    # a and b are learnable scalars
    from src.common.tensors.abstraction import AbstractTensor
    if not hasattr(learnable_metric_tensor_func, "a"):
        learnable_metric_tensor_func.a = AbstractTensor.tensor(1.0, requires_grad=True)
        learnable_metric_tensor_func.b = AbstractTensor.tensor(0.1, requires_grad=True)
    I = AbstractTensor.eye(3).reshape(1, 1, 1, 3, 3)
    uvw = AbstractTensor.stack([u, v, w], dim=-1).reshape(u.shape + (3,))
    outer = uvw.unsqueeze(-1) * uvw.unsqueeze(-2)  # (...,3,3)
    g = learnable_metric_tensor_func.a * I + learnable_metric_tensor_func.b * outer
    g_inv = g  # For demo, just use g as its own inverse
    det = AbstractTensor.ones(u.shape)
    return g, g_inv, det

def main():
    grid_shape = (5, 5, 5)
    in_channels = 2
    out_channels = 2
    transform = RectangularTransform(Lx=1.0, Ly=1.0, Lz=1.0, device="cpu")
    wrapper = MetricSteeredConv3DWrapper(
        in_channels=in_channels,
        out_channels=out_channels,
        grid_shape=grid_shape,
        transform=transform,
        boundary_conditions=("dirichlet",) * 6,
        k=3,
        eig_from="g",
        pointwise=True,
        deploy_mode="modulated",
        laplace_kwargs={"metric_tensor_func": learnable_metric_tensor_func, "lambda_reg": 0.5},
    )
    print("[INFO] Created MetricSteeredConv3DWrapper with grid_shape:", grid_shape)
    x = AbstractTensor.randn((1, in_channels, *grid_shape), device="cpu")
    print("[INFO] Input tensor shape:", x.shape)
    y = wrapper.forward(x)
    print("[INFO] Forward pass output shape:", y.shape)
    # Add LSN regularization loss to main loss
    lsn = wrapper.local_state_network
    reg_loss = getattr(lsn, "_regularization_loss", None)
    if reg_loss is not None:
        print(f"[INFO] LSN regularization loss: {float(reg_loss.item())}")
        loss = y.sum() + reg_loss
    else:
        print("[WARN] No LSN regularization loss found, using only output sum.")
        loss = y.sum()
    print("[INFO] Total loss (output + reg):", float(loss.item()))
    loss.backward()
    # Call LSN backward with grads of weighted/modulated outputs if available
    grad_w = getattr(lsn, "_weighted_padded", None)
    grad_m = getattr(lsn, "_modulated_padded", None)
    if grad_w is not None and grad_m is not None:
        grad_w = getattr(grad_w, "_grad", None) or AbstractTensor.zeros_like(lsn._weighted_padded)
        grad_m = getattr(grad_m, "_grad", None) or AbstractTensor.zeros_like(lsn._modulated_padded)
        lsn.backward(grad_w, grad_m, lambda_reg=0.5)
    print("[INFO] Backward pass complete. Checking gradients...")
    # Collect all parameters: wrapper, LSN, and learnable metric
    all_params = list(wrapper.parameters(include_structural=True))
    # Add metric function learnable params
    for name in ["a", "b"]:
        p = getattr(learnable_metric_tensor_func, name, None)
        if p is not None:
            all_params.append(p)
    all_ok = True
    for param in all_params:
        grad = getattr(param, "grad", None)
        if grad is None:
            grad = getattr(param, "_grad", None)
        label = getattr(param, "_label", str(param))
        if grad is None:
            print(f"[FAIL] {label}: grad is None")
            all_ok = False
        elif not (grad.abs() > 0).any():
            print(f"[FAIL] {label}: grad is all zero")
            all_ok = False
        else:
            print(f"[OK]   {label}: grad present, min={float(grad.min().item()):.3g}, max={float(grad.max().item()):.3g}, mean={float(grad.mean().item()):.3g}")
    if all_ok:
        print("[SUCCESS] All parameters received nonzero gradients.")
    else:
        print("[ERROR] Some parameters did not receive gradients.")

if __name__ == "__main__":
    main()
