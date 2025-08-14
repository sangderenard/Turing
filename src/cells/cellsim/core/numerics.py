from tqdm.auto import tqdm  # type: ignore


def clamp_nonneg(x: float, eps: float = 1e-18) -> float:
    """Clamp negative values to zero without inserting a floor concentration.

    Previously this helper returned ``eps`` for non-positive inputs which meant
    that a compartment with ``n=0`` moles and ``V=0`` volume would report a
    concentration of 1 mol/m³.  That artificial osmotic term could flip the sign
    of fluxes in edge cases.  Returning ``0.0`` keeps quantities non-negative
    without sneaking in extra solute.
    """
    return x if x > 0.0 else 0.0

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
    for _ in tqdm(range(max_iter), desc="imex", leave=False):
        y_prev = y_new
        y_new = y + dt * (f_explicit(y) + f_implicit(y_new))
        if abs(y_new - y_prev) < tol:
            break
    return y_new
