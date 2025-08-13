
import numpy as np
from .constraints import PlaneContact

class XPBDSolver:
    def __init__(self, params):
        self.p = params

    def integrate(self, X, V, invm, dt):
        V[:] *= self.p.damping
        X[:] += dt * V

    def build_contacts(self, X, box_min, box_max):
        contacts = []
        xmin,ymin,zmin = box_min
        xmax,ymax,zmax = box_max
        for i,x in enumerate(X):
            contacts.append(PlaneContact(i, np.array([ 1.0, 0.0, 0.0]), -xmin, self.p.contact_compliance)) # x>=xmin
            contacts.append(PlaneContact(i, np.array([-1.0, 0.0, 0.0]),  xmax, self.p.contact_compliance)) # -x+xmax>=0 -> x<=xmax
            contacts.append(PlaneContact(i, np.array([ 0.0, 1.0, 0.0]), -ymin, self.p.contact_compliance))
            contacts.append(PlaneContact(i, np.array([ 0.0,-1.0, 0.0]),  ymax, self.p.contact_compliance))
            contacts.append(PlaneContact(i, np.array([ 0.0, 0.0, 1.0]), -zmin, self.p.contact_compliance))
            contacts.append(PlaneContact(i, np.array([ 0.0, 0.0,-1.0]),  zmax, self.p.contact_compliance))
        return contacts

    def project(self, constraints, X, invm, faces, vol_func, vol_grads_func, dt, iters, box_min, box_max):
        for _ in range(iters):
            for c in constraints.get("stretch", []):
                c.project(X, invm, dt)
            for c in constraints.get("bending", []):
                c.project(X, invm, dt)
            vc = constraints.get("volume", None)
            if vc is not None:
                vc.project(X, invm, faces, vol_func, vol_grads_func, dt)
            for cnt in self.build_contacts(X, box_min, box_max):
                cnt.project(X, invm, dt)
