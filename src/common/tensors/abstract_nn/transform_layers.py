"""
transform_layers.py
-------------------

Lightweight reshaping helpers that prepare tensors for 3D convolution.

The layers exposed here are intentionally parameter free and rely on
`AbstractTensor` for their tensor manipulations so that they integrate with the
existing autograd tape.  They convert 2‑D or 3‑D inputs into the canonical
`(B, C, D, H, W)` layout expected by the convolutional demos.
"""

from ..abstraction import AbstractTensor as AT


class Transform2DLayer:
    """Lift ``(B, H, W)`` tensors to ``(B, C=1, D=1, H, W)``."""

    def parameters(self):  # pragma: no cover - no trainable params
        return []

    def forward(self, x):
        xt = AT.get_tensor(x)
        if xt.ndim != 3:
            raise ValueError(f"expected 3D input (B,H,W), got {xt.shape}")
        return xt[:, None, None, :, :]

    def get_input_shape(self):
        return (None, None, None)


class Transform3DLayer:
    """Insert a channel axis so ``(B, D, H, W)`` → ``(B, C=1, D, H, W)``."""

    def parameters(self):  # pragma: no cover - no trainable params
        return []

    def forward(self, x):
        xt = AT.get_tensor(x)
        if xt.ndim != 4:
            raise ValueError(f"expected 4D input (B,D,H,W), got {xt.shape}")
        return xt[:, None, ...]

    def get_input_shape(self):
        return (None, None, None, None)
