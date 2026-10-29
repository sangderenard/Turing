import math


# The scientific contract (transcendentals: signal_cores) on every scalar
# math op that has a core: exp, log, atan, tanh, the ``**`` operator and
# math.pow.  ir_identities.lower_transcendentals_to_signal_cores rewrites
# each into WideExpansionEmitter SSA, so C, LLVM and Fortran compute the
# same correctly rounded doubles; before, LLVM refused atan ("operation has
# no repository LLVM emission") and each lane used its host library.
def root(x: float, y: float) -> float:
    a = math.exp(y) + math.log(x)
    b = math.atan(x) + math.tanh(y)
    c = x ** y + math.pow(x, -y)
    return 1.0 * a + 2.0 * b + 4.0 * c
# INPUTS [{'x': 0.5, 'y': 1.5}, {'x': 3.0, 'y': -0.25}, {'x': 1e-3, 'y': 20.0}, {'x': 1.0000000000000002, 'y': 700.0}, {'x': 7.25, 'y': 0.0}]
