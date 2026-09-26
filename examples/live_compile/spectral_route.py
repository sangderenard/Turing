def spectral_route(left, right, phase):
    mixed = (left + right) * 0.5
    carrier = mixed.sin() + phase.cos()
    energy = carrier * carrier
    return energy / (1.0 + energy)
