"""Opt-in immutable geometry capture using the authored NumPy-f32 tensor law.

The dynamic manifold remains studio_mesh_manifold.query. No particle equation
is duplicated or reassociated. GPU transcendental bit parity is not asserted:
static atan2/sqrt are explicitly evaluated on the existing NumPy-f32 backend.
Changing any validated vertex requires another embedding/admission and bake.
"""
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import numpy as np
from src.common.tensors import AbstractTensor as T


def _freeze(value):
    if isinstance(value,tuple):return tuple(_freeze(v) for v in value)
    a=np.asarray(value.data)
    if a.size!=1 or not np.isfinite(a).all():
        raise ValueError("static feature geometry must be a finite scalar/vector")
    return float(a.item())


def _lift(value,zero):
    # Use the incumbent wall-normal span-promotion idiom. The supported scalar
    # tensor_like path was measured too, but emits more bookkeeping here.
    return tuple(_lift(v,zero) for v in value) if isinstance(value,tuple) else zero+value


@dataclass(frozen=True)
class StaticGeometryBake:
    boundary_digest: str
    records: object
    coefficient_digest: str
    backend: str="AbstractTensor NumPy float32"

    def value(self,key,produce,zero):
        return _lift(self.records[key],zero)

    def to_data(self):
        return {"schema":"turing.static-feature-geometry.v1","boundary_digest":self.boundary_digest,
                "coefficient_digest":self.coefficient_digest,"backend":self.backend,
                "numpy_version":np.__version__,"gpu_transcendental_bit_parity_verified":False,
                "values":[{"identity":list(key),"value":value} for key,value in self.records.items()]}


def bake_static_geometry(boundary):
    """Capture before tracing; returned immutable values belong to one digest."""
    records={}
    class Capture:
        boundary_digest=boundary.digest
        def value(self,key,produce,zero):
            value=produce()
            if key in records:raise ValueError("duplicate static geometry identity")
            records[key]=_freeze(value)
            return value
    zero=T.get_tensor(np.asarray([0],dtype=np.float32))
    boundary.query((zero,zero,zero),static_geometry=Capture())
    digest=hashlib.sha256(repr((boundary.digest,np.__version__,tuple(records.items()))).encode()).hexdigest()
    return StaticGeometryBake(boundary.digest,MappingProxyType(records),digest)
