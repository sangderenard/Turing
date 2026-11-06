import math


def root(e: float, n: float, c: float, radius: float):
    # ``k = radius * arc / (rho + 1e-12)`` with ``arc`` a ``math`` function's
    # result: nothing proved the class of that result, so the ``Div`` that
    # read it refused ("region value N (Div) has no settled dtype"); the
    # ``Mul`` beside it settled through the tensor-operation path only.  The
    # patch projection of a spherical game's shot, with ``sin`` for ``atan2``
    # (the C lane has no atan2 spelling; the Wasm test of the real law,
    # test_wasm_core_product.py, divides an ``atan2``).
    rho = math.sqrt(e * e + n * n)
    arc = math.sin(rho + c)
    k = radius * arc / (rho + 1e-12)
    return e * k, n * k, math.sqrt(c * c + 1.0) / (n * n + 1.0)

# INPUTS [{'e': 0.3, 'n': 0.4, 'c': 0.8, 'radius': 30.0}, {'e': -0.7, 'n': 0.1, 'c': -0.2, 'radius': 30.0}, {'e': 0.0, 'n': 0.0, 'c': 1.0, 'radius': 30.0}, {'e': 1e-9, 'n': -2e-9, 'c': 1.0, 'radius': 7.5}]
