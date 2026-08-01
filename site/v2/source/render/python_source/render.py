
def interest_network(unit_x, unit_y, interest):
    # Frozen network source enters the same AOT/WASM pipeline as the fractal.
    h0 = (unit_x * 0.83 + unit_y * -0.41 + interest * 0.72 + 0.15).tanh()
    h1 = (unit_x * -0.36 + unit_y * 0.91 + interest * 0.48 - 0.08).tanh()
    return (h0 * 0.62 + h1 * -0.57 + interest * 0.31).tanh()


def shade(count):
    # mandelbrot_jpeg_planes, ported: sqrt of the normalised escape count,
    # three cosine channels offset by 0, 0.21 and 0.43, scaled to 0-255 and
    # clamped, exactly as the original composed its display planes.
    #
    # The original raised each channel to 1.65. WebAssembly has no pow
    # instruction, and reaching it through exp/log would need log below the
    # quarter its table starts at -- which is where a colour ramp spends most
    # of its range. x^1.625 = x * sqrt(x) * sqrt(sqrt(sqrt(x))) uses only
    # sqrt and multiply, both native, and differs from x^1.65 by under two
    # percent across [0,1] -- below a quantisation step at 8 bits.
    phase = (count * 0.00625).minimum(1.0).maximum(0.0).sqrt()

    wave_r = ((phase + 0.0) * 6.283185307179586).cos() * 0.5 + 0.5
    r_half = wave_r.sqrt()
    red = (wave_r * r_half * r_half.sqrt().sqrt() * 255.0 + 0.5).minimum(255.0).maximum(0.0)

    wave_g = ((phase + 0.21) * 6.283185307179586).cos() * 0.5 + 0.5
    g_half = wave_g.sqrt()
    green = (wave_g * g_half * g_half.sqrt().sqrt() * 255.0 + 0.5).minimum(255.0).maximum(0.0)

    wave_b = ((phase + 0.43) * 6.283185307179586).cos() * 0.5 + 0.5
    b_half = wave_b.sqrt()
    blue = (wave_b * b_half * b_half.sqrt().sqrt() * 255.0 + 0.5).minimum(255.0).maximum(0.0)

    # A tuple return does not lower, so one channel is returned and all three
    # remain as executed steps for the build to name as outputs.
    return red + green * 0.0 + blue * 0.0


def quadratic_family(unit_x, unit_y, center_x, center_y, span,
                     family_mix, julia_x, julia_y):
    # The continuous Mandelbrot-to-Julia quadratic family. family_mix = 0 is
    # the Mandelbrot set (orbit starts at zero, constant is the pixel); 1 is
    # a Julia set (orbit starts at the pixel, constant is fixed); everything
    # between is a real member of the family rather than a cross-fade of two
    # pictures.
    cx = center_x + unit_x * span
    cy = center_y + unit_y * span
    zx = cx * family_mix
    zy = cy * family_mix
    constant_x = cx + family_mix * (julia_x - cx)
    constant_y = cy + family_mix * (julia_y - cy)
    count = cx * 0.0
    clamp_value = cx * 0.0 + 1e+18
    for _ in range(160):
        zx2 = zx * zx
        zy2 = zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        next_zx = zx2 - zy2 + constant_x
        next_zy = 2.0 * zx * zy + constant_y
        zx = next_zx.minimum(clamp_value).maximum(-clamp_value)
        zy = next_zy.minimum(clamp_value).maximum(-clamp_value)
    return count


def render(unit_x, unit_y, t, interest):
    # The page supplies the grid and the clock. Everything else -- where the
    # camera is, how deep, which member of the family -- is computed here.
    #
    # sin, cos and exp2 have no WebAssembly instruction; they arrive as
    # bounded lookup tables baked into this module, which is what lets the
    # whole trajectory live in the compiled program instead of being worked
    # out in JavaScript and fed in.
    #
    # Written as separate assignments rather than a tuple-returning helper:
    # simultaneous tuple assignment does not lower (see aot_compile).
    #
    # animated_camera + dream_parameters from demo_mandelbrot_fusion,
    # ported. The audio terms are the only omission -- the original modulated
    # three of these with bass/low_mid/high_mid, and reaction = 0 removes
    # exactly those. zoom_rate is 0 by default, so there is no progressive
    # dive: the span oscillates about the base and returns.
    #
    # t is TRAVEL, not a frame count; every frequency here is in travel units.
    log_zoom = (t * 0.71).sin() * 1.25 + (t * 1.93).sin() * 0.45
    mandelbrot_span = log_zoom.exp() * 0.004
    dx = ((t * 0.83).sin() * 0.58 + (t * 2.17).sin() * 0.22) * 0.004
    dy = ((t * 0.97 + 0.61).sin() * 0.48 + (t * 1.67).sin() * 0.19) * 0.004 - 0.0011
    mandelbrot_center_x = dx - 0.743643887
    mandelbrot_center_y = dy + 0.131825904

    # family_mix stays in [0.04, 0.22]: larger excursions need a different
    # camera chart and erase a deep Mandelbrot view's structure.
    family_mix = ((t * 0.24).sin() * 0.5 + 0.5) * 0.18 + 0.04

    # c = mu/2 - mu^2/4 parameterises the main cardioid; |mu| < 1 keeps the
    # Julia sets connected rather than dust.
    mu_x = (t * 0.31).cos() * 0.58
    mu_y = (t * 0.31).sin() * 0.58
    mu2_x = mu_x * mu_x - mu_y * mu_y
    mu2_y = mu_x * mu_y * 2.0
    julia_x = mu_x * 0.5 - mu2_x * 0.25
    julia_y = mu_y * 0.5 - mu2_y * 0.25

    # Preserve the target c-plane exactly under the family transform:
    # (1-mix)*pixel + mix*julia == mandelbrot_pixel.
    family_scale = family_mix * -1.0 + 1.0
    center_x = (mandelbrot_center_x - julia_x * family_mix) / family_scale
    center_y = (mandelbrot_center_y - julia_y * family_mix) / family_scale
    span = mandelbrot_span / family_scale
    drift = interest_network(unit_x=unit_x, unit_y=unit_y, interest=interest)
    return shade(quadratic_family(
        unit_x=unit_x + drift * 0.004,
        unit_y=unit_y + drift * -0.003,
        center_x=center_x, center_y=center_y, span=span,
        family_mix=family_mix, julia_x=julia_x, julia_y=julia_y,
    ))