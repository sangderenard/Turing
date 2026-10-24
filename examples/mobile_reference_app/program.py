def mobile_main(phase: float, touch_x: float, touch_y: float):
    """Public application program consumed by every deployment shell."""
    red = 0.15 + phase * 0.55
    green = 0.20 + touch_x * 0.65
    blue = 0.25 + touch_y * 0.65
    return red, green, blue
