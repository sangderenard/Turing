def clamp_nonneg(x: float, eps: float=1e-18) -> float:
    return x if x > eps else eps

def adapt_dt(dt: float, rel: float) -> float:
    if rel > 1e-3:   # too big: halve
        return max(dt*0.5, 1e-12)
    if rel < 1e-5:   # small: relax slightly
        return min(dt*1.1, 1.0)
    return dt
