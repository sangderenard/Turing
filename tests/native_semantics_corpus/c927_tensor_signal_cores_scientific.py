import numpy as np


# The scientific contract (transcendentals: signal_cores) on elementwise
# float64 tensor math: np.exp, np.log, np.sin, np.cos, np.tan, np.tanh and
# the ``**`` operator on an array.  Each was a lowercase tensor op that C
# spelled with libm per element, LLVM with its unary_double kernel and
# Fortran with an elemental intrinsic; ir_identities.
# scalarize_elementwise_transcendentals now makes each one a loop whose
# body is the scalar signal core, so every element is the correctly
# rounded double on all three backends.  Hard arguments: exp(192.92...)
# (glibc 1 ulp off, c913), exp(534.7...), sin/cos(1e22), log(1 + ulp).
# Every element stays inside Python's math domain: the per-element guards
# fault where math.exp/log/sin would raise (np.exp(1e22) would be inf).
def root(x: float, y: float):
    a = np.asarray([x, 534.707910443053, 0.75])
    e = np.exp(a)
    b = np.asarray([y, 3.0, 0.75])
    s = np.sin(b) + np.cos(b)
    t = np.tan(b) + np.tanh(b)
    c = np.asarray([x * 0.0 + 1.0000000000000002, 2.5, 0.75])
    g = np.log(c) + c ** 1.5
    # Each element its own returned leaf, so each is judged on its own.
    return (1.0 * e[0], 1.0 * e[1], 1.0 * s[0], 1.0 * t[2], 1.0 * g[0], 1.0 * g[1])
# INPUTS [{'x': 192.9234597167931, 'y': 1e22}, {'x': 0.5, 'y': 3.0}, {'x': -740.5, 'y': 1.5707963267948966}, {'x': 7.5e-17, 'y': -0.25}]
