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
        return self.model.forward(x)
