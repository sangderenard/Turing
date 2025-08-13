
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

    def project(self, constraints, X, invm, faces, vol_func, vol_grads_func, dt, iters, box_min, box_max):
        for _ in range(iters):
            for c in constraints.get("stretch", []):
                c.project(X, invm, dt)
            for c in constraints.get("bending", []):
                c.project(X, invm, dt)
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
