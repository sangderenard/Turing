from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.tensors import AbstractTensor
from src.common.tensors.linalg import solve

rng = np.random.default_rng(7)
worst = 0.0
worst_case = None
failures = []
tested = 0
for trial in range(400):
    n = int(rng.integers(2, 6))
    # small integers make exact ties in |pivot| common
    a = rng.integers(-2, 3, size=(n, n)).astype(np.float64)
    b = rng.integers(-3, 4, size=n).astype(np.float64)
    if abs(np.linalg.det(a)) < 1e-8:
        continue
    tested += 1
    expected = np.linalg.solve(a, b)
    try:
        produced = np.asarray(
            solve(AbstractTensor.get_tensor(a.tolist()),
                  AbstractTensor.get_tensor(b.tolist())).tolist(),
            dtype=np.float64,
        ).reshape(expected.shape)
    except Exception as error:
        failures.append((n, a.tolist(), b.tolist(), f"{type(error).__name__}: {error}"))
        continue
    scale = max(1.0, float(np.max(np.abs(expected))))
    err = float(np.max(np.abs(produced - expected))) / scale
    if not np.isfinite(err) or err > 1e-8:
        failures.append((n, a.tolist(), b.tolist(), f"err={err:.3e}"))
    if np.isfinite(err) and err > worst:
        worst, worst_case = err, (n, a.tolist(), b.tolist())

print("tested", tested, "failures", len(failures), flush=True)
print("worst relative error", "%.3e" % worst, flush=True)
for item in failures[:5]:
    print("FAIL n=%d %s" % (item[0], item[3]), flush=True)
    print("   A", item[1], flush=True)
    print("   b", item[2], flush=True)

batch_a = rng.normal(size=(2, 3, 3))
batch_b = rng.normal(size=(2, 3))
try:
    produced = np.asarray(
        solve(AbstractTensor.get_tensor(batch_a.tolist()),
              AbstractTensor.get_tensor(batch_b.tolist())).tolist(),
        dtype=np.float64,
    )
    expected = np.linalg.solve(batch_a, batch_b[..., None])[..., 0]
    err = float(np.max(np.abs(produced.reshape(expected.shape) - expected)))
    print("batched 2x(3x3) max_err=%.3e" % err,
          "OK" if err < 1e-9 else "MISMATCH", flush=True)
except Exception as error:
    print("batched 2x(3x3) RAISED", type(error).__name__, error, flush=True)
