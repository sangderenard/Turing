from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.tensors import AbstractTensor
from src.common.tensors.linalg import det, inv, solve, eye, trace, norm, dot

def run(label, fn, expected, scale_by_expected=True):
    try:
        produced = np.asarray(fn(), dtype=np.float64)
    except Exception as error:
        return (label, f"RAISED {type(error).__name__}: {error}")
    expected = np.asarray(expected, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(expected)))) if scale_by_expected else 1.0
    err = float(np.max(np.abs(produced.reshape(expected.shape) - expected))) / scale
    if not np.isfinite(err) or err > 1e-8:
        return (label, f"err={err:.3e}")
    return None

rng = np.random.default_rng(11)
failures = []
counts = {"det": 0, "inv": 0, "trace": 0, "norm": 0, "dot": 0}
for trial in range(120):
    n = int(rng.integers(2, 6))
    a = rng.integers(-2, 3, size=(n, n)).astype(np.float64)
    if abs(np.linalg.det(a)) < 1e-8:
        continue
    at = AbstractTensor.get_tensor(a.tolist())
    counts["det"] += 1
    item = run(f"det n={n}", lambda: det(at).tolist(), np.linalg.det(a))
    if item: failures.append((item, a.tolist()))
    counts["inv"] += 1
    item = run(f"inv n={n}", lambda: inv(at).tolist(), np.linalg.inv(a))
    if item: failures.append((item, a.tolist()))
    counts["trace"] += 1
    item = run(f"trace n={n}", lambda: trace(at).tolist(), np.trace(a))
    if item: failures.append((item, a.tolist()))
    v = rng.normal(size=n)
    vt = AbstractTensor.get_tensor(v.tolist())
    counts["norm"] += 1
    item = run(f"norm n={n}", lambda: norm(vt).tolist(), np.linalg.norm(v))
    if item: failures.append((item, v.tolist()))
    w = rng.normal(size=n)
    wt = AbstractTensor.get_tensor(w.tolist())
    counts["dot"] += 1
    item = run(f"dot n={n}", lambda: dot(vt, wt).tolist(), float(v @ w))
    if item: failures.append((item, [v.tolist(), w.tolist()]))

item = run("eye(4)", lambda: eye(4).tolist(), np.eye(4))
if item: failures.append((item, None))

print("checked", counts, flush=True)
print("failures", len(failures), flush=True)
for (label, reason), data in failures[:6]:
    print("FAIL", label, reason, flush=True)
    print("   ", data, flush=True)
