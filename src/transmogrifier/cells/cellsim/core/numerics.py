def clamp_nonneg(x: float, eps: float=1e-18) -> float:
    return x if x > eps else eps

def adapt_dt(dt: float, rel: float) -> float:
    if rel > 1e-3:   # too big: halve
        return max(dt*0.5, 1e-12)
    if rel < 1e-5:   # small: relax slightly
        return min(dt*1.1, 1.0)
    return dt


def imex_euler(y, f_explicit, f_implicit, dt, max_iter=8, tol=1e-8):
    """A minimal IMEX (implicit-explicit) Euler step.

    Returns updated state solving y_{n+1} = y_n + dt*(f_explicit(y_n) + f_implicit(y_{n+1})).
    Uses fixed-point iteration; suitable for light stiffness when electrochem/CRN enabled.
    """
    y_new = y + dt * f_explicit(y)
    if f_implicit is None:
        return y_new
    for _ in range(max_iter):
        y_prev = y_new
        y_new = y + dt * (f_explicit(y) + f_implicit(y_new))
        if abs(y_new - y_prev) < tol:
            break
    return y_new
