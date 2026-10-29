import math


# The compute contract (transcendentals: backend_intrinsic): run with
# TYPED_CHECK_NUMERIC_MODE=compute and each lane spells these with its own
# math library -- C libm, LLVM llvm.* and libm calls (atan was refused:
# "operation has no repository LLVM emission"), gfortran's SIN/COS (the
# default 1e-6 sine table returned c22 4.8e-8 off).  Under the default
# scientific contract the same program takes the signal cores.
def root(x: float) -> float:
    s = math.sin(x) + math.cos(x) + math.tan(x)
    e = math.exp(x) + math.log(x + 2.0)
    return 1.0 * s + 2.0 * e + 4.0 * math.atan(x) + 8.0 * math.tanh(x) + x ** 0.75
# INPUTS [{'x': 0.5}, {'x': 1.25}, {'x': 3.0}, {'x': 10.5}]
