from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.riemann.grid_block import RiemannGridBlock


def _example_config():
    return {
        "geometry": {
            "key": "rect_euclidean",
            "grid_shape": (2, 2, 2),
            "boundary_conditions": (True,) * 6,
            "transform_args": {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
            "laplace_kwargs": {},
        },
        "pre_linear": {"in_dim": 2, "out_dim": 3},
        "film": {"in_dim": 3, "out_dim": 3},
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


def test_forward_output_shape():
    cfg = _example_config()
    block = RiemannGridBlock.build_from_config(cfg)
    B = 1
    D, H, W = cfg["geometry"]["grid_shape"]
    x = AbstractTensor.randn((B, cfg["pre_linear"]["in_dim"], D, H, W))
    y = block.forward(x)
    assert y.shape == (B, cfg["post_linear"]["out_dim"], D, H, W)


def test_parameters_registered():
    cfg = _example_config()
    block = RiemannGridBlock.build_from_config(cfg)
    params = block.parameters()
    expected = 8  # pre(2) + film(2) + conv(2) + post(2)
    assert len(params) == expected
    for p in params:
        assert isinstance(p, AbstractTensor)


def test_forward_end_to_end():
    cfg = _example_config()
    block = RiemannGridBlock.build_from_config(cfg)
    D, H, W = cfg["geometry"]["grid_shape"]
    x = AbstractTensor.ones((1, cfg["pre_linear"]["in_dim"], D, H, W))
    y = block.forward(x)
    assert y.shape == (1, cfg["post_linear"]["out_dim"], D, H, W)
