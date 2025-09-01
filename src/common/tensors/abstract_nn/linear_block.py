"""
linear_block.py
----------------

A simple linear block model for testing and training.
"""

from ..abstraction import AbstractTensor as AT
from ..abstract_nn.core import Linear, Model
from .activations import GELU, Sigmoid

class LinearBlock:
    """Flexible three-layer linear adapter stack.

    The block accepts arbitrary input and output sizes.  A hidden dimension is
    chosen as the integer mean of ``input_dim`` and ``output_dim`` when not
    explicitly provided.  Three ``Linear`` layers are chained so the block can
    expand or contract between mismatched endpoints.

    Parameters
    ----------
    input_dim : int
        Size of the incoming feature dimension.
    output_dim : int
        Desired output feature size.
    like : AT
        AbstractTensor-like object used for parameter initialisation.
    hidden_dim : int, optional
        Manually override the intermediate feature size.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        like: AT,
        hidden_dim: int | None = None,
    ):
        if hidden_dim is None:
            hidden_dim = int((input_dim + output_dim) / 2)
        self.model = Model(
            layers=[
                Linear(input_dim, hidden_dim, like=like, init="xavier"),
                Linear(hidden_dim, hidden_dim, like=like, init="xavier"),
                Linear(hidden_dim, output_dim, like=like, init="xavier"),
            ],
            activations=[GELU(), GELU(), Sigmoid()],
        )

    def parameters(self):
        return self.model.parameters()
    def forward(self, x):

        # Infer the adapter I/O once so we can route shapes correctly.
        in_dim = int(self.model.layers[0].W.shape[0])
        out_dim = int(self.model.layers[-1].W.shape[1])

        # Helper: robust shape tuple for AbstractTensor / numpy-backed
        shape = x.shape() if callable(getattr(x, "shape", None)) else x.shape
        ndim = len(shape)

        # Case A: already 2D (N, in_dim) — just run the MLP.
        if ndim == 2 and int(shape[-1]) == in_dim:
            return self.model.forward(x)

        # Case B: last axis is the feature axis (..., in_dim) — flatten leading dims.
        if int(shape[-1]) == in_dim:
            # Collapse everything but the last (=features) axis into batch.
            batch = 1
            for s in shape[:-1]:
                batch *= int(s)
            y2 = self.model.forward(x.reshape((batch, in_dim)))
            return y2.reshape((*shape[:-1], out_dim))

        # Case C: channels-first tensor where channel axis = 1 (e.g., B,C,*,*,*).
        if ndim >= 3 and int(shape[1]) == in_dim:
            B = int(shape[0]); C = int(shape[1])
            # Flatten spatial dims to one axis.
            spatial = 1
            for s in shape[2:]:
                spatial *= int(s)
            # (B, C, S) -> (B*S, C) so Linear sees C as features.
            xs = x.reshape((B, C, spatial)).swapaxes(1, 2).reshape((B * spatial, C))
            ys = self.model.forward(xs)  # (B*S, out_dim)
            # Restore to (B, out_dim, *spatial_shape)
            y = ys.reshape((B, spatial, out_dim)).swapaxes(1, 2).reshape((B, out_dim, *shape[2:]))
            return y

        else:
            raise ValueError(f"Unexpected input shape {shape}")

