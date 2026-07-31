"""Value-computing observer for target-neutral compiler discovery.

The compiler needs concrete scalar values while it observes source control,
but the observation pass must not compile code for the eventual target.
This backend therefore uses NumPy storage and algorithms while disabling
backend conveniences whose public AbstractTensor methods have a canonical
primitive decomposition.  The recorded tape describes that portable
decomposition, not NumPy's optional high-level shortcut.
"""

from __future__ import annotations

import numpy as np

from ..abstraction import register_backend
from ..numpy_backend import NumPyTensorOperations


class PrecompileObserverTensorOperations(NumPyTensorOperations):
    """NumPy execution with the canonical compilable operation surface."""

    def _torch_dtype_to_numpy(self, dtype):
        mapped = super()._torch_dtype_to_numpy(dtype)
        if mapped is not None or dtype is None:
            return mapped
        try:
            return np.dtype(dtype)
        except TypeError:
            return None

    def to_dtype_(self, dtype="float"):
        mapped = self._torch_dtype_to_numpy(dtype)
        if mapped is not None:
            return self.data.astype(mapped)
        return super().to_dtype_(dtype)

    def clamp_(self, min_val=None, max_val=None):
        # AbstractTensor.clamp then records maximum/minimum, matching the
        # primitive path used by backends without a native clamp shortcut.
        raise NotImplementedError


register_backend("precompile_observer", PrecompileObserverTensorOperations)


__all__ = ["PrecompileObserverTensorOperations"]
