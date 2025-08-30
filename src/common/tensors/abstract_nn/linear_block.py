"""
linear_block.py
----------------

A simple linear block model for testing and training.
"""

from ..abstraction import AbstractTensor as AT
from ..abstract_nn.core import Linear, Model
from .activations import GELU, Sigmoid

class LinearBlock:
    """
    A simple linear block model that can be used as a standalone network.

    Args:
        input_dim (int): The size of the input dimension.
        hidden_dim (int): The size of the hidden layer.
        output_dim (int): The size of the output dimension.
        like (AT): AbstractTensor-like object for tensor creation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, like: AT):
        self.model = Model(
            layers=[
                Linear(input_dim, hidden_dim, like=like, init="xavier"),
                Linear(hidden_dim, output_dim, like=like, init="xavier"),
            ],
            activations=[GELU(), Sigmoid()]
        )

    def parameters(self):
        return self.model.parameters()

    def forward(self, x):
        return self.model.forward(x)
