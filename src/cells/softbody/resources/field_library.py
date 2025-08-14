# field_library.py
import numpy as np
from src.cells.softbody.engine.fields import VectorField

def gravity(g=(0.0, -9.81, 0.0), selector=None, dim=3):
    g = np.asarray(g, dtype=float)
    return VectorField(fn=lambda X,t,c,w: g, units="accel", selector=selector, dim=dim)

def uniform_flow(u=(0.05, 0.0, 0.0), selector=None, dim=3):
    u = np.asarray(u, dtype=float)
    return VectorField(fn=lambda X,t,c,w: u, units="velocity", selector=selector, dim=dim)

def shear_flow(rate=0.1, axis_xy=(0,1), selector=None, dim=3):
    i, j = axis_xy  # e.g., (0,1): u_x = rate * y
    def fn(X, t, c, w):
        U = np.zeros_like(X)
        U[:, i] = rate * X[:, j]
        return U
    return VectorField(fn=fn, units="velocity", selector=selector, dim=dim)

def solid_body_vortex(omega=0.5, center=(0.0,0.0), plane=(0,1), selector=None, dim=3):
    px, py = plane
    cx, cy = center
    def fn(X, t, c, w):
        U = np.zeros_like(X)
        dx = X[:, px] - cx
        dy = X[:, py] - cy
        U[:, px] = -omega * dy
        U[:, py] =  omega * dx
        return U
    return VectorField(fn=fn, units="velocity", selector=selector, dim=dim)

def membrane_potential(phi: callable, k=1.0, h=1e-3, selector=None, dim=3):
    """
    Acceleration a = -k ∇phi(X). If phi returns (N,) potentials given X (N,D).
    Finite-difference gradient so you can drop any scalar field.
    """
    def fn(X, t, c, w):
        N, D = X.shape
        grad = np.zeros_like(X)
        for d in range(D):
            e = np.zeros((1, D)); e[0, d] = 1.0
            xp = X + h*e; xm = X - h*e
            # phi must accept (N,D) and return (N,)
            grad[:, d] = (phi(xp, t, c, w) - phi(xm, t, c, w)) / (2*h)
        return -k * grad
    return VectorField(fn=fn, units="accel", selector=selector, dim=dim)

def fluid_noise(sigma=1e-3, com_neutral=False, selector=None, dim=3, rng=np.random):
    """
    Displacement Δx ~ N(0, (sigma^2 * dt) I) — we'll scale by sqrt(dt) in caller.
    We encode scaling inside fn as it doesn’t know dt; we pass dt via world.dt.
    """
    def fn(X, t, c, w):
        dt = getattr(w, "dt", 0.0)
        n = rng.standard_normal(X.shape) * (sigma * np.sqrt(max(dt, 0.0)))
        if com_neutral:
            n -= n.mean(axis=0, keepdims=True)
        return n
    return VectorField(fn=fn, units="displacement", selector=selector, dim=dim)
