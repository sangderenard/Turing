
import numpy as np

class XPBDSolver:
    def __init__(self, params):
        self.p = params

    def integrate(self, X, V, invm, dt):
        V[:] *= self.p.damping
        X[:] += dt * V

    def build_contacts(self, X, box_min, box_max):
        """Return vertices violating the axis-aligned bounding box.

        The classic implementation created ``PlaneContact`` objects per vertex
        and plane.  Here we compute signed distances against all six planes in a
        single vectorised pass and return the indices, normals and penetration
        depths for contacts that actually violate the box.
        """
        normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        d = np.array(
            [
                -box_min[0],
                box_max[0],
                -box_min[1],
                box_max[1],
                -box_min[2],
                box_max[2],
            ],
            dtype=np.float64,
        )
        C = X @ normals.T + d  # (n_verts, 6)
        mask = C < 0.0
        if not np.any(mask):
            return np.array([], dtype=int), np.empty((0, 3)), np.array([])
        vidx, plane_idx = np.nonzero(mask)
        return vidx, normals[plane_idx], C[vidx, plane_idx]

    def _project_stretch(self, X, invm, cons, dt):
        idx = cons["indices"]
        if idx.size == 0:
            return
        i = idx[:, 0]
        j = idx[:, 1]
        xi = X[i]
        xj = X[j]
        d = xj - xi
        L = np.linalg.norm(d, axis=1)
        mask = L > 1e-12
        n = np.zeros_like(d)
        n[mask] = d[mask] / L[mask][:, None]
        C = L - cons["rest"]
        wi = invm[i]
        wj = invm[j]
        wsum = wi + wj
        mask &= wsum > 0.0
        if not np.any(mask):
            return
        alpha = cons["compliance"] / (dt * dt)
        dl = np.zeros_like(cons["lamb"])
        dl[mask] = -(C[mask] + alpha[mask] * cons["lamb"][mask]) / (
            wsum[mask] + alpha[mask]
        )
        cons["lamb"] += dl
        dp = n * dl[:, None]
        np.add.at(X, i, -wi[:, None] * dp)
        np.add.at(X, j, wj[:, None] * dp)

    def _project_bending(self, X, invm, cons, dt):
        idx = cons["indices"]
        if idx.size == 0:
            return
        i, j, k, l = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        pi, pj, pk, pl = X[i], X[j], X[k], X[l]
        n1 = np.cross(pk - pi, pj - pi)
        n2 = np.cross(pj - pl, pi - pl)
        n1_norm = np.linalg.norm(n1, axis=1)
        n2_norm = np.linalg.norm(n2, axis=1)
        mask = (n1_norm > 1e-12) & (n2_norm > 1e-12)
        n1[mask] /= n1_norm[mask][:, None]
        n2[mask] /= n2_norm[mask][:, None]
        cos_th = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
        th = np.arccos(cos_th)
        C = th - cons["rest"]
        e = pj - pi
        e_len = np.linalg.norm(e, axis=1)
        mask &= e_len > 1e-12
        e_hat = np.zeros_like(e)
        e_hat[mask] = e[mask] / e_len[mask][:, None]
        denom1 = np.linalg.norm(np.cross(pk - pi, pj - pi), axis=1) + 1e-12
        denom2 = np.linalg.norm(np.cross(pl - pj, pi - pj), axis=1) + 1e-12
        grad_i = np.cross(n1, e_hat) / denom1[:, None] + np.cross(n2, e_hat) / denom2[:, None]
        grad_j = -np.cross(n1, e_hat) / denom1[:, None] - np.cross(n2, e_hat) / denom2[:, None]
        grad_k = np.cross(e_hat, n1) / denom1[:, None]
        grad_l = np.cross(e_hat, n2) / denom2[:, None]
        w = np.stack([invm[i], invm[j], invm[k], invm[l]], axis=1)
        grads = np.stack([grad_i, grad_j, grad_k, grad_l], axis=1)
        denom = np.sum(w[:, :, None] * (grads * grads), axis=(1, 2))
        mask &= denom > 1e-12
        if not np.any(mask):
            return
        alpha = cons["compliance"] / (dt * dt)
        dl = np.zeros_like(cons["lamb"])
        dl[mask] = -(C[mask] + alpha[mask] * cons["lamb"][mask]) / (
            denom[mask] + alpha[mask]
        )
        cons["lamb"] += dl
        dP = w[:, :, None] * (dl[:, None, None] * grads)
        np.add.at(X, i, dP[:, 0])
        np.add.at(X, j, dP[:, 1])
        np.add.at(X, k, dP[:, 2])
        np.add.at(X, l, dP[:, 3])

    def project(self, constraints, X, invm, faces, vol_func, vol_grads_func, dt, iters, box_min, box_max):
        for _ in range(iters):
            sc = constraints.get("stretch")
            if sc is not None:
                self._project_stretch(X, invm, sc, dt)
            bc = constraints.get("bending")
            if bc is not None:
                self._project_bending(X, invm, bc, dt)
            vc = constraints.get("volume", None)
            if vc is not None:
                vc.project(X, invm, faces, vol_func, vol_grads_func, dt)

            # Vectorised contact projection against the bounding box
            idx, normals, C = self.build_contacts(X, box_min, box_max)
            if len(idx):
                w = invm[idx]
                mask = w > 0.0
                if np.any(mask):
                    idx = idx[mask]
                    normals = normals[mask]
                    C = C[mask]
                    w = w[mask]
                    alpha = self.p.contact_compliance / (dt * dt)
                    dl = -C / (w + alpha)
                    X[idx] += (w * dl)[:, None] * normals
