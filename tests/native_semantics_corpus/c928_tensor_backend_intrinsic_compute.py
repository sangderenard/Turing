import numpy as np


# The compute contract (transcendentals: backend_intrinsic): run with
# TYPED_CHECK_NUMERIC_MODE=compute and each lane keeps the elementwise
# tensor op on its own math library -- C libm per element, LLVM's
# unary_double kernel, Fortran's elemental intrinsics.  Under the default
# scientific contract the same program takes the signal cores (c927).
def root(x: float) -> float:
    a = np.asarray([x, x * 0.5, 0.75])
    e = np.exp(a) + np.log(a + 2.0)
    s = np.sin(a) + np.cos(a) + np.tanh(a)
    return 1.0 * e[0] + 2.0 * e[1] + 4.0 * s[2] + 8.0 * s[0]
# INPUTS [{'x': 0.5}, {'x': 1.25}, {'x': 3.0}, {'x': 10.5}]
