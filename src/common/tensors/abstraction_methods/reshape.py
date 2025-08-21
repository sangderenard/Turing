from __future__ import annotations

from typing import Any, Tuple
def reshape(self, shape: int) -> "AbstractTensor":
    """Return a reshaped tensor as an AbstractTensor, using the backend registry pattern."""
    if hasattr(self, 'reshape_'):
        return self.reshape_(shape)
    raise NotImplementedError("Reshape fallback not implemented for this backend.")

def transpose(self, dim0: int = 0, dim1: int = 1) -> "AbstractTensor":
    """Return a transposed tensor as an AbstractTensor, using the backend registry pattern."""
    if hasattr(self, 'transpose_'):
        return self.transpose_(dim0, dim1)
    raise NotImplementedError("Transpose fallback not implemented for this backend.")

def squeeze(self, dim: int | None = None) -> "AbstractTensor":
    """Return a tensor with all (or one) dimensions of size 1 removed."""
    if hasattr(self, 'squeeze_'):
        return self.squeeze_(dim)
    raise NotImplementedError("Squeeze fallback not implemented for this backend.")

def reshape(self, shape: int) -> "AbstractTensor":
    """Return a reshaped tensor as an AbstractTensor, using the backend registry pattern."""
    if hasattr(self, 'reshape_'):
        return self.reshape_(shape)
    try:
        from ..abstraction import BACKEND_REGISTRY
        backend_cls = BACKEND_REGISTRY.get("numpy")
        if backend_cls is not None:
            numpy_tensor = backend_cls(track_time=getattr(self, 'track_time', False))
            numpy_tensor = numpy_tensor.ensure_tensor(self.data)
            if hasattr(numpy_tensor.data, 'reshape'):
                reshaped_data = numpy_tensor.data.reshape(shape)
                reshaped_tensor = backend_cls(track_time=getattr(self, 'track_time', False))
                reshaped_tensor.data = reshaped_data
                return reshaped_tensor.to_backend(self)
    except Exception:
        pass
    # Fallback: pure python backend (not implemented here)
    raise NotImplementedError("Reshape fallback not implemented for pure python backend.")

def flatten(self):
    """Return a flattened version of the tensor as an AbstractTensor, using the backend registry pattern."""
    if hasattr(self, 'flatten_'):
        return self.flatten_()
    try:
        from ..abstraction import BACKEND_REGISTRY
        backend_cls = BACKEND_REGISTRY.get("numpy")
        if backend_cls is not None:
            numpy_tensor = backend_cls(track_time=getattr(self, 'track_time', False))
            numpy_tensor = numpy_tensor.ensure_tensor(self.data)
            if hasattr(numpy_tensor.data, 'flatten'):
                flat_data = numpy_tensor.data.flatten()
                flat_tensor = backend_cls(track_time=getattr(self, 'track_time', False))
                flat_tensor.data = flat_data
                return flat_tensor.to_backend(self)
    except Exception:
        pass
    # Fallback: pure python backend if available
    try:
        from ..abstraction import BACKEND_REGISTRY
        backend_cls = BACKEND_REGISTRY.get("pure_python")
        if backend_cls is not None:
            py_tensor = backend_cls(track_time=getattr(self, 'track_time', False))
            py_tensor = py_tensor.ensure_tensor(self.data)
            def _flatten(data):
                if not isinstance(data, list):
                    return [data]
                return [item for sublist in data for item in _flatten(sublist)]
            flat_data = _flatten(py_tensor.data)
            flat_tensor = backend_cls(track_time=getattr(self, 'track_time', False))
            flat_tensor.data = flat_data
            return flat_tensor.to_backend(self)
    except Exception:
        pass
    def _flatten(data):
        if not isinstance(data, list):
            return [data]
        return [item for sublist in data for item in _flatten(sublist)]
    flat_data = _flatten(self.data)
    out = type(self)(track_time=getattr(self, 'track_time', False))
    out.data = flat_data
    return out

def repeat(self, repeats: Any = None, dim: int = 0) -> "AbstractTensor":
    """Repeat ``self`` along ``dim`` ``repeats`` times."""
    return self.repeat_(repeats, dim)
def repeat_interleave(
        self, repeats: int = 1, dim: Optional[int] = None
    ) -> "AbstractTensor":
        result = AbstractTensor.get_tensor(self.repeat_interleave_(repeats, dim))
        return result
