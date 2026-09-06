"""Small feedback route for the compile/profile/trace spectral-dye demo."""


def spectral_dye_trace_demo(left, right, steps):
    total = left
    carry = right
    for _ in range(steps):
        total = total * 0.83 + carry
        carry = carry * 0.91 + total * 0.07
    return total + carry
