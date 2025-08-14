
import numpy as np
from dataclasses import dataclass

# --------- Stretch (edge length) ---------
@dataclass
class StretchConstraint:
    i: int; j: int
    rest: float
    compliance: float = 0.0
    lamb: float = 0.0

    def project(self, X, invm, dt):
        xi = X[self.i]; xj = X[self.j]
        d = xj - xi
        L = np.linalg.norm(d)
        if L < 1e-12: return
        n = d / L
        C = L - self.rest
        wi = invm[self.i]; wj = invm[self.j]
        wsum = wi + wj
        if wsum <= 0: return
        alpha = self.compliance / (dt*dt)
        dl = -(C + alpha*self.lamb) / (wsum + alpha)
        self.lamb += dl
        dp = dl * n
        X[self.i] -= wi * dp
        X[self.j] += wj * dp

# --------- Volume (closed surface) ---------
@dataclass
class VolumeConstraint:
    target: float
    compliance: float = 0.0
    lamb: float = 0.0

    def project(self, X, invm, faces, volume_func, volume_grads_func, dt):
        V = volume_func(X, faces)
        C = V - self.target
        grads = volume_grads_func(X, faces)
        denom = float(np.sum(invm[:,None] * (grads*grads)))
        if denom <= 1e-16: return
        alpha = self.compliance / (dt*dt)
        dl = -(C + alpha*self.lamb) / (denom + alpha)
        self.lamb += dl
        X += invm[:,None] * (dl * grads)

# --------- Bending (dihedral around edge) ---------
def _triangle_normal(a,b,c):
    n = np.cross(b-a, c-a)
    ln = np.linalg.norm(n)
    return n/ln if ln>1e-12 else n

@dataclass
class DihedralBendingConstraint:
    i: int; j: int; k: int; l: int
    rest_angle: float
    compliance: float = 0.0
    lamb: float = 0.0

    def project(self, X, invm, dt):
        i,j,k,l = self.i, self.j, self.k, self.l
        pi, pj, pk, pl = X[i], X[j], X[k], X[l]
        n1 = _triangle_normal(pi, pk, pj)
        n2 = _triangle_normal(pj, pl, pi)
        if np.linalg.norm(n1)==0 or np.linalg.norm(n2)==0: return
        cos_th = np.clip(np.dot(n1, n2), -1.0, 1.0)
        th = np.arccos(cos_th)
        C = th - self.rest_angle

        e = pj - pi
        e_len = np.linalg.norm(e)
        if e_len < 1e-12: return
        e_hat = e / e_len

        denom1 = np.linalg.norm(np.cross(pk - pi, pj - pi)) + 1e-12
        denom2 = np.linalg.norm(np.cross(pl - pj, pi - pj)) + 1e-12

        grad_i =  np.cross(n1, e_hat)/denom1 + np.cross(n2, e_hat)/denom2
        grad_j = -np.cross(n1, e_hat)/denom1 - np.cross(n2, e_hat)/denom2
        grad_k =  np.cross(e_hat, n1)/denom1
        grad_l =  np.cross(e_hat, n2)/denom2

        w = np.array([invm[i], invm[j], invm[k], invm[l]])
        grads = np.stack([grad_i, grad_j, grad_k, grad_l], axis=0)
        denom = float(np.sum(w[:,None]*(grads*grads)))
        if denom <= 1e-12: return
        alpha = self.compliance / (dt*dt)
        dl = -(C + alpha*self.lamb) / (denom + alpha)
        self.lamb += dl
        dP = (w[:,None] * (dl * grads))
        X[i] += dP[0]; X[j] += dP[1]; X[k] += dP[2]; X[l] += dP[3]

# --------- Contact vs box planes ---------
@dataclass
class PlaneContact:
    vidx: int
    n: np.ndarray
    d: float
    compliance: float = 0.0
    lamb: float = 0.0

    def project(self, X, invm, dt):
        x = X[self.vidx]
        C = np.dot(self.n, x) + self.d  # want >= 0
        if C >= 0:
            self.lamb = 0.0
            return
        w = invm[self.vidx]
        if w <= 0: return
        alpha = self.compliance / (dt*dt)
        dl = -(C + alpha*self.lamb) / (w + alpha)
        self.lamb += dl
        X[self.vidx] += (w * dl) * self.n
