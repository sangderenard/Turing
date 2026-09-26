from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.tensors import AbstractTensor
from src.common.tensors.linalg import solve

def check(label, a, b):
    at = AbstractTensor.get_tensor(a.tolist())
    bt = AbstractTensor.get_tensor(b.tolist())
    try:
        produced = np.asarray(solve(at, bt).tolist(), dtype=np.float64)
    except Exception as error:
        print(f"{label:28s} RAISED {type(error).__name__}: {error}", flush=True)
        return
    expected = np.linalg.solve(a, b)
    err = float(np.max(np.abs(produced.reshape(expected.shape) - expected)))
    print(f"{label:28s} max_err={err:.3e} {'OK' if err < 1e-9 else 'MISMATCH'}",
          flush=True)
    if err >= 1e-9:
        print("   produced", produced.reshape(-1), flush=True)
        print("   expected", expected.reshape(-1), flush=True)

rng = np.random.default_rng(0)
check("2x2 simple", np.array([[4.,1.],[2.,3.]]), np.array([1.,2.]))
check("2x2 needs pivot", np.array([[0.,1.],[2.,3.]]), np.array([1.,2.]))
check("3x3 random", rng.normal(size=(3,3)), rng.normal(size=3))
check("4x4 random", rng.normal(size=(4,4)), rng.normal(size=4))
check("5x5 random", rng.normal(size=(5,5)), rng.normal(size=5))
check("3x3 tied |pivot| same sign", np.array([[1.,2.,3.],[1.,0.,1.],[0.,1.,4.]]),
      np.array([1.,2.,3.]))
check("3x3 tied |pivot| opp sign", np.array([[1.,2.,3.],[-1.,0.,1.],[0.,1.,4.]]),
      np.array([1.,2.,3.]))
check("3x3 three-way tie", np.array([[1.,2.,3.],[1.,0.,1.],[1.,1.,5.]]),
      np.array([1.,2.,3.]))
check("2x2 rhs matrix", np.array([[4.,1.],[2.,3.]]),
      np.array([[1.,0.],[2.,1.]]))
check("6x6 random", rng.normal(size=(6,6)), rng.normal(size=6))
check("4x4 near-singular", np.array([[1.,2.,3.,4.],[2.,4.1,6.,8.],
                                     [3.,6.,9.2,12.],[4.,8.,12.,16.4]]),
      np.array([1.,2.,3.,4.]))
batch_a = rng.normal(size=(2,3,3))
batch_b = rng.normal(size=(2,3))
label = "batched 2x(3x3)"
at = AbstractTensor.get_tensor(batch_a.tolist())
bt = AbstractTensor.get_tensor(batch_b.tolist())
try:
    produced = np.asarray(solve(at, bt).tolist(), dtype=np.float64)
    expected = np.linalg.solve(batch_a, batch_b)
    err = float(np.max(np.abs(produced.reshape(expected.shape) - expected)))
    print(label, "max_err=%.3e" % err, "OK" if err < 1e-9 else "MISMATCH", flush=True)
except Exception as error:
    print(label, "RAISED", type(error).__name__, error, flush=True)
