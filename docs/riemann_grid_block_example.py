"""Minimal demonstration of RiemannGridBlock configuration."""

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.riemann.grid_block import RiemannGridBlock

cfg = {
    "geometry": {
        "key": "rect_euclidean",
        "grid_shape": (2, 2, 2),
        "boundary_conditions": (True,) * 6,
    },
    "casting": {
        "mode": "pre_linear",
        "film": {"enabled": True},
        "coords": {"type": "raw", "dims": 3},
    },
    "conv": {
        "in_channels": 3,
        "out_channels": 4,
        "k": 1,
        "metric_source": "g",
        "boundary_conditions": ("dirichlet",) * 6,
        "pointwise": True,
    },
    "post_linear": {"in_dim": 4, "out_dim": 2},
}

block = RiemannGridBlock.build_from_config(cfg)

D, H, W = cfg["geometry"]["grid_shape"]
x = AbstractTensor.randn((1, cfg["conv"]["in_channels"], D, H, W))
y = block.forward(x)
print("output shape:", y.shape)
