from __future__ import annotations

"""
SpectralConv3D (MVP)
-------------------

LB‑spectral convolution with learned multipliers over a small set of eigenmodes.
Implements a simple, backend‑agnostic pipeline using AbstractTensor ops.

Notes
- Assumes manifold provides dense LB eigenpairs (evals, evecs) for a small grid.
- No mass matrix weighting is applied (basic Euclidean inner product projection).
"""

from typing import Any
from ..abstraction import AbstractTensor
from ..autograd import autograd


class SpectralConv3D:
    def __init__(self, in_channels: int, out_channels: int, *, num_modes: int = 16) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_modes = int(num_modes)
        # Learnable spectral multipliers: (C_out, C_in, K)
        scale = (2.0 / max(1, self.in_channels * self.num_modes)) ** 0.5
        w = [[[AbstractTensor.random.gauss(0.0, scale) for _ in range(self.num_modes)]
              for _ in range(self.in_channels)]
             for _ in range(self.out_channels)]
        self.Wspec = AbstractTensor.tensor_from_list(w, requires_grad=True, tape=autograd.tape)
        autograd.tape.create_tensor_node(self.Wspec)
        self.Wspec._label = "SpectralConv3D.Wspec"
        autograd.tape.annotate(self.Wspec, label=self.Wspec._label)

    def parameters(self):
        return [self.Wspec]

    def zero_grad(self):
        if hasattr(self.Wspec, "zero_grad"):
            self.Wspec.zero_grad()

    def forward(self, x: Any, *, manifold) -> Any:
        """
        x: (B, C, D, H, W)
        manifold: ManifoldPackage with (evals, evecs)
        """
        B, C, D, H, W = x.shape
        N = D * H * W
        # Ensure eigenpairs are available
        eps = manifold.eigenpairs()
        if eps is None:
            # Build with the requested number of modes if not already present
            manifold.num_eigenpairs = max(manifold.num_eigenpairs or 0, self.num_modes)
            manifold.build()
            eps = manifold.eigenpairs()
        assert eps is not None, "Manifold did not provide eigenpairs"
        evals, evecs = eps  # evals: (K,), evecs: (N, K)
        autograd.tape.annotate(evals, label="SpectralConv3D.evals")
        autograd.tape.annotate(evecs, label="SpectralConv3D.evecs")
        K = min(self.num_modes, int(evecs.shape[1]))
        E = evecs[:, :K]              # (N, K)
        Wspec = self.Wspec[:, :, :K]  # (C_out, C_in, K)
        autograd.tape.annotate(E, label="SpectralConv3D.E_modes")
        autograd.tape.annotate(Wspec, label="SpectralConv3D.Wspec_slice")

        # Flatten spatial and project: alpha = x_flat @ E  → (B, C, K)
        x_flat = x.reshape(B, C, N)            # (B, C, N)
        alpha = x_flat @ E                     # (B, C, K)
        autograd.tape.annotate(x_flat, label="SpectralConv3D.x_flat")
        autograd.tape.annotate(alpha, label="SpectralConv3D.alpha")

        # Apply spectral weights: Beta[b, co, k] = sum_ci alpha[b, ci, k] * Wspec[co, ci, k]
        a_exp = alpha.reshape(B, 1, C, K)                      # (B, 1, C, K)
        W_exp = Wspec.reshape(1, self.out_channels, C, K)      # (1, C_out, C, K)
        Beta = (a_exp * W_exp).sum(dim=2)                      # (B, C_out, K)
        autograd.tape.annotate(Beta, label="SpectralConv3D.Beta")

        # Reconstruct: y_flat[b, co, :] = Beta[b, co, :] @ E^T
        y_flat = Beta @ E.swapaxes(0, 1)                       # (B, C_out, N)
        y = y_flat.reshape(B, self.out_channels, D, H, W)
        autograd.tape.annotate(y, label="SpectralConv3D.output")
        return y
