# RiemannGridBlock Configuration

`RiemannGridBlock` combines a geometry package with a metric‑steered
convolution and optional casting/regularisation stages.  Instances are usually
constructed from a configuration dictionary passed to
`RiemannGridBlock.build_from_config`.

## Schema

The configuration uses the following keys.  Unspecified optional fields fall
back to the defaults listed below.

### `geometry` *(required)*

```python
{
    "key": str,                    # selects builder in geometry_factory
    "grid_shape": (int, int, int) = (1, 1, 1),
    "boundary_conditions": (bool, bool, bool, bool, bool, bool) = (True,) * 6,
    "transform_args": dict = {},
    "laplace_kwargs": dict = {},
}
```

### `conv` *(required)*

```python
{
    "in_channels": int,
    "out_channels": int,
    "k": int = 3,
    "metric_source": str = "g",
    "boundary_conditions": (str, str, str, str, str, str) = ("dirichlet",) * 6,
    "pointwise": bool = True,
    "stencil": {
        "offsets": sequence[int] = (-1, 0, 1),
        "length": int = 1,
        "normalize": bool = False,
    },
}
```

### `casting` *(optional)*

```python
{
    "mode": "pre_linear" | "fixed" | "soft_assign" = "fixed",
    "film": {"enabled": bool} = {"enabled": False},
    "coords": {"type": "raw"|"fourier", "dims": int} | None = None,
    "inject_coords": bool = False,
    "map": "row_major" | "1to1" | "normalized_span" = "row_major",
}
```

### `post_linear` *(optional)*

```python
{"in_dim": int, "out_dim": int}
```

### `regularization` *(optional)*

```python
{
    "smooth_bins": float | int,
    "weight_decay": {"pre": float, "conv": float, "post": float},
}
```

## Example

```python
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
```

A runnable version of this example is provided in
[`riemann_grid_block_example.py`](riemann_grid_block_example.py).
