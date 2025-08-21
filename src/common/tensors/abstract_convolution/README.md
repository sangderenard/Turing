
# abstract_convolution (initial drop)

This module packages the **advanced Laplace–Beltrami builder** and the **LocalStateNetwork**
as a foundation for ND/manifold convolutions.

- `laplace_nd.py` — modern, vectorized 3D Laplace builder (from `laplace2.py`), suitable for spectral/heat kernels.
- `local_state_network.py` — the metric-tensor network that produces per-voxel state (metric, inverse, det, fields).

## Running tests

```bash
pytest -q abstract_convolution/tests
```
