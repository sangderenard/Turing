"""
linear_layer_demo.py
--------------------

Minimal demo: trains a single AbstractTensor Linear layer on a synthetic
regression mapping (y = X @ W_true + b_true), with explicit autograd
annotations and sanity checks on parameter gradients. This reduces
complexity relative to the convolutional Riemann demo to isolate and
verify the autograd/optimizer interaction on one layer.
"""

from __future__ import annotations

from ..abstraction import AbstractTensor
from ..autograd import autograd
from .core import Linear, RectConv2d, RectConv3d
from .optimizer import Adam
from ..logger import get_tensors_logger


logger = get_tensors_logger()


def build_synthetic_regression(N: int = 256, in_dim: int = 4, out_dim: int = 3):
    """Create a simple linear regression dataset on the AbstractTensor backend.

    Returns
    -------
    X : AbstractTensor  (N, in_dim)
    Y : AbstractTensor  (N, out_dim)
    W_true, b_true : ground-truth parameters
    """
    AT = AbstractTensor
    # Inputs
    X = AT.randn((N, in_dim), requires_grad=True)
    autograd.tape.annotate(X, label="linear_demo.X")
    autograd.tape.auto_annotate_eval(X)

    # Ground-truth parameters (fixed, no grad)
    W_true = AT.randn((in_dim, out_dim), requires_grad=False)
    b_true = AT.randn((1, out_dim), requires_grad=False) * 0.1
    autograd.tape.annotate(W_true, label="linear_demo.W_true")
    autograd.tape.annotate(b_true, label="linear_demo.b_true")
    autograd.tape.auto_annotate_eval(W_true)
    autograd.tape.auto_annotate_eval(b_true)

    # Targets: y = X @ W_true + b_true
    Y = X @ W_true
    b_broadcast = b_true.broadcast_rows(X.shape[0], label="linear_demo.b_true.broadcast")
    Y = Y + b_broadcast
    autograd.tape.annotate(Y, label="linear_demo.Y")
    autograd.tape.auto_annotate_eval(Y)
    return X, Y, W_true, b_true


def _collect_params_and_grads(layer: Linear):
    params = list(layer.parameters())
    grads = [getattr(p, "grad", None) for p in params]
    for i, (p, g) in enumerate(zip(params, grads)):
        label = getattr(p, "_label", None)
        logger.info(
            f"Param {i}: label={label}, shape={getattr(p, 'shape', None)}, grad is None={g is None}, grad shape={getattr(g, 'shape', None) if g is not None else None}"
        )
    return params, grads


def demo_linear():
    AT = AbstractTensor
    N, in_dim, out_dim = 256, 4, 3
    X, Y, W_true, b_true = build_synthetic_regression(N=N, in_dim=in_dim, out_dim=out_dim)

    # Build a single Linear layer on the same backend
    like = AT.get_tensor()  # pick current/available backend
    layer = Linear(in_dim=in_dim, out_dim=out_dim, like=like, init="xavier")

    # Loss and optimizer
    def mse(a, b):
        return ((a - b) ** 2).mean()

    params, _ = _collect_params_and_grads(layer)
    optimizer = Adam(params, lr=1e-2)

    # Train until convergence or max epochs
    for epoch in range(1, 5001):
        # Clear (tape-level) grads on parameters
        for p in layer.parameters():
            if hasattr(p, "zero_grad"):
                p.zero_grad()
            elif hasattr(p, "grad") and p.grad is not None:
                p._grad = AT.zeros_like(p.grad)

        # Forward
        pred = layer.forward(X)
        autograd.tape.annotate(pred, label="linear_demo.pred")
        autograd.tape.auto_annotate_eval(pred)

        # Loss
        loss = mse(pred, Y)
        autograd.tape.annotate(loss, label="linear_demo.loss")
        autograd.tape.auto_annotate_eval(loss)

        # Backward through the tape, accumulating grads on current parameters
        if hasattr(loss, "backward"):
            loss.backward()

        # Re-collect params and grads and run basic checks
        params, grads = _collect_params_and_grads(layer)
        for p in params:
            label = getattr(p, "_label", None)
            assert hasattr(p, "grad"), f"Parameter {label or p} has no grad attribute"
            assert p.grad is not None, f"Parameter {label or p} grad is None after backward()"
            assert p.grad.shape == p.shape, (
                f"Parameter {label or p} has incorrect grad shape: grad.shape={getattr(p.grad, 'shape', None)}, param.shape={getattr(p, 'shape', None)}"
            )
        for g, p in zip(grads, params):
            label = getattr(p, "_label", None)
            assert hasattr(g, "shape"), f"Gradient for {label or p} has no shape attribute"
            assert g.shape == p.shape, (
                f"Gradient for {label or p} has incorrect shape: grad.shape={getattr(g, 'shape', None)}, param.shape={getattr(p, 'shape', None)}"
            )

        # Optimizer step: update in-place to keep parameter identity on the tape
        new_params = optimizer.step(params, grads)
        for p, new_p in zip(params, new_params):
            AbstractTensor.copyto(p, new_p)

        if epoch % 100 == 0 or float(loss.item()) < 1e-6:
            print(f"Epoch {epoch}: loss={float(loss.item()):.2e}")
        if float(loss.item()) < 1e-6:
            print("Converged.")
            break

    # Final audit: show learned vs. true parameter norms (coarse check)
    W_learned, b_learned = layer.W, layer.b
    try:
        wn = float(((W_learned * W_learned).sum()).sqrt().item())
        wt = float(((W_true * W_true).sum()).sqrt().item())
        print(f"||W_learned||={wn:.4f} vs ||W_true||={wt:.4f}")
        if b_learned is not None:
            bn = float(((b_learned * b_learned).sum()).sqrt().item())
            bt = float(((b_true * b_true).sum()).sqrt().item())
            print(f"||b_learned||={bn:.4f} vs ||b_true||={bt:.4f}")
    except Exception:
        pass


def _collect_params_and_grads_generic(layer):
    params = list(layer.parameters())
    grads = [getattr(p, "grad", None) for p in params]
    for i, (p, g) in enumerate(zip(params, grads)):
        label = getattr(p, "_label", None)
        logger.info(
            f"Param {i}: label={label}, shape={getattr(p, 'shape', None)}, grad is None={g is None}, grad shape={getattr(g, 'shape', None) if g is not None else None}"
        )
    return params, grads


def demo_conv2d():
    AT = AbstractTensor
    # Shapes
    N, Cin, Cout, H, W = 8, 2, 3, 16, 16
    kH = kW = 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    # Inputs
    X = AT.randn((N, Cin, H, W), requires_grad=True)

    # Ground truth weights (fixed)
    W_true = AT.randn((Cout, Cin, kH, kW), requires_grad=False)
    b_true = AT.randn((Cout,), requires_grad=False) * 0.1

    # Target via im2col conv: Y = (Wm_true @ cols) + b
    cols = X.unfold2d((kH, kW), stride=stride, padding=padding, dilation=dilation)
    Wm_true = W_true.reshape(Cout, Cin * kH * kW)
    Y_cols = Wm_true @ cols  # (N, Cout, L)
    Y_cols = Y_cols + b_true.reshape(1, -1, 1)
    # Output spatial size
    pH, pW = padding
    sH, sW = stride
    dH, dW = dilation
    Hout = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    Wout = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    Y = Y_cols.reshape(N, Cout, Hout, Wout)

    # Train a conv2d layer to match Y
    like = AT.get_tensor()
    layer = RectConv2d(Cin, Cout, (kH, kW), stride=stride, padding=padding, dilation=dilation, like=like, bias=True)
    mse = lambda a, b: ((a - b) ** 2).mean()
    params, _ = _collect_params_and_grads_generic(layer)
    opt = Adam(params, lr=1e-2)

    for epoch in range(1, 2001):
        # Zero grads
        for p in layer.parameters():
            if hasattr(p, "zero_grad"):
                p.zero_grad()
            elif hasattr(p, "grad") and p.grad is not None:
                p._grad = AT.zeros_like(p.grad)
        # Forward
        pred = layer.forward(X)
        loss = mse(pred, Y)
        # Compute grads explicitly for current layer parameters to avoid any
        # dependency on the tape's global parameter registry.
        _params = list(layer.parameters())
        autograd.grad(loss, _params, retain_graph=False, allow_unused=False)
        # Checks
        params, grads = _collect_params_and_grads_generic(layer)
        for p in params:
            assert p.grad is not None and p.grad.shape == p.shape
        # Update in place
        new_params = opt.step(params, grads)
        for p, np_ in zip(params, new_params):
            AbstractTensor.copyto(p, np_)
        if epoch % 100 == 0 or float(loss.item()) < 1e-6:
            print(f"[conv2d] Epoch {epoch}: loss={float(loss.item()):.2e}")
        if float(loss.item()) < 1e-6:
            print("[conv2d] Converged.")
            break


def demo_conv3d_pointwise():
    AT = AbstractTensor
    # Shapes: 1x1x1 conv acts as per-voxel linear map across channels
    N, Cin, Cout, D, H, W = 4, 2, 3, 4, 6, 6
    kD = kH = kW = 1
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)

    X = AT.randn((N, Cin, D, H, W), requires_grad=True)
    W_true = AT.randn((Cout, Cin, kD, kH, kW), requires_grad=False)
    b_true = AT.randn((Cout,), requires_grad=False) * 0.1

    # Target: pointwise linear projection at each voxel
    X_flat = X.reshape(N, Cin, D * H * W)
    Wm_true = W_true.reshape(Cout, Cin)
    Y_flat = Wm_true @ X_flat  # (N, Cout, L)
    Y_flat = Y_flat + b_true.reshape(1, -1, 1)
    Y = Y_flat.reshape(N, Cout, D, H, W)

    like = AT.get_tensor()
    layer = RectConv3d(Cin, Cout, (kD, kH, kW), stride=stride, padding=padding, dilation=dilation, like=like, bias=True)
    mse = lambda a, b: ((a - b) ** 2).mean()
    params, _ = _collect_params_and_grads_generic(layer)
    opt = Adam(params, lr=1e-2)

    for epoch in range(1, 2001):
        for p in layer.parameters():
            if hasattr(p, "zero_grad"):
                p.zero_grad()
            elif hasattr(p, "grad") and p.grad is not None:
                p._grad = AT.zeros_like(p.grad)
        pred = layer.forward(X)
        loss = mse(pred, Y)
        _params = list(layer.parameters())
        autograd.grad(loss, _params, retain_graph=False, allow_unused=False)
        params, grads = _collect_params_and_grads_generic(layer)
        for p in params:
            assert p.grad is not None and p.grad.shape == p.shape
        new_params = opt.step(params, grads)
        for p, np_ in zip(params, new_params):
            AbstractTensor.copyto(p, np_)
        if epoch % 100 == 0 or float(loss.item()) < 1e-6:
            print(f"[conv3d] Epoch {epoch}: loss={float(loss.item()):.2e}")
        if float(loss.item()) < 1e-6:
            print("[conv3d] Converged.")
            break


def main():
    print("Running linear demo...")
    demo_linear()
    print("Running conv2d demo...")
    demo_conv2d()
    print("Running conv3d (pointwise) demo...")
    demo_conv3d_pointwise()


if __name__ == "__main__":
    main()
