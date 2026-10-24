# Two returned names bound to one value (the tire's membrane_face returns
# pressure_force_0_x_n and pressure_force_1_x_n as one sympy value).  The
# second name had no output slot and the function_output page refused.
def root(n: float):
    p = n * 2.0
    a = p
    b = p
    return a, b
# INPUTS [{'n': 1.5}, {'n': -2.0}, {'n': 0.0}]
