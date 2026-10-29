# ``drive(advance, t)`` calls its ``step`` parameter, as the managed tire's
# ``run_superstep`` calls ``balloon_tire_managed_advance``.  ``advance`` is
# reached only through the parameter; pursuit must still mark it active so
# its own call to ``scale`` carries reachability authority.
from src.common.tensors.abstraction import AbstractTensor


def scale(t):
    return t * 2.0


def advance(t):
    return scale(t) + 0.5


def drive(step, t):
    return step(t)


def root(n: float) -> float:
    t = AbstractTensor.tensor(n)
    out = drive(advance, t)
    return float(out.item())
# INPUTS [{'n': 0.0}, {'n': 1.5}, {'n': -2.25}]
