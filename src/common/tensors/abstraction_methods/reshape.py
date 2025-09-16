from __future__ import annotations

from typing import Any, Optional


def _wrap_result(self, result):
    from ..abstraction import AbstractTensor
    if isinstance(result, AbstractTensor):
        return result
    out = type(self)(track_time=getattr(self, "track_time", False))
    out.data = result
    return out


def _reshape_dispatch(self, shape_tuple, op_name: str, fallback_error: str) -> "AbstractTensor":
    from ..abstraction import AbstractTensor, BACKEND_REGISTRY

    if hasattr(self, "reshape_"):
        out = _wrap_result(self, self.reshape_(shape_tuple))
        finalize = AbstractTensor._pre_autograd(
            op_name, [self], params={"new_shape": getattr(out, "shape", shape_tuple)}
        )
        return finalize(out)

    # numpy fallback (kept intact, but now passes a tuple)
    try:
        backend_cls = BACKEND_REGISTRY.get("numpy")
        if backend_cls is not None:
            numpy_tensor = backend_cls(track_time=getattr(self, "track_time", False))
            numpy_tensor = numpy_tensor.ensure_tensor(self.data)
            if hasattr(numpy_tensor.data, "reshape"):
                reshaped_data = numpy_tensor.data.reshape(shape_tuple)
                reshaped_tensor = backend_cls(track_time=getattr(self, "track_time", False))
                reshaped_tensor.data = reshaped_data
                finalize = AbstractTensor._pre_autograd(
                    op_name, [self], params={"new_shape": reshaped_tensor.shape}
                )
                converted = reshaped_tensor.to_backend(self)
                return finalize(converted)
    except Exception:
        pass

    raise NotImplementedError(fallback_error)


def reshape(self, *shape) -> "AbstractTensor":
    """Return a reshaped tensor as an AbstractTensor."""
    from ..abstraction import AbstractTensor

    shape_tuple = AbstractTensor._normalize_shape_args(*shape)
    return _reshape_dispatch(
        self,
        shape_tuple,
        op_name="reshape",
        fallback_error="Reshape fallback not implemented for pure python backend.",
    )


def view(self, *shape) -> "AbstractTensor":
    """Alias for :meth:`reshape` mirroring PyTorch's ``view`` semantics."""
    from ..abstraction import AbstractTensor

    shape_tuple = AbstractTensor._normalize_shape_args(*shape)
    return _reshape_dispatch(
        self,
        shape_tuple,
        op_name="view",
        fallback_error="View fallback not implemented for pure python backend.",
    )

def transpose(self, dim0: int = 0, dim1: int = 1) -> "AbstractTensor":
    """Return a transposed tensor as an AbstractTensor."""
    from ..abstraction import AbstractTensor

    shape = getattr(self, "shape", ())
    ndim = len(shape)
    if not (-ndim <= dim0 < ndim) or not (-ndim <= dim1 < ndim):
        raise ValueError("dim0 or dim1 out of range")
    dim0 %= ndim
    dim1 %= ndim

    if hasattr(self, "transpose_"):
        perm = list(range(ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        finalize = AbstractTensor._pre_autograd(
            "permute", [self], params={"perm": perm}
        )
        return finalize(_wrap_result(self, self.transpose_(dim0, dim1)))
    raise NotImplementedError("Transpose fallback not implemented for this backend.")


def permute(self, *dims: int) -> "AbstractTensor":
    """Return a tensor with its dimensions permuted according to ``dims``."""
    from ..abstraction import AbstractTensor, BACKEND_REGISTRY

    perm = dims[0] if len(dims) == 1 and isinstance(dims[0], (list, tuple)) else dims
    shape = getattr(self, "shape", ())
    if len(perm) != len(shape):
        raise ValueError("permute requires dims to match tensor dimensions")

    if hasattr(self, "permute_"):
        result = _wrap_result(self, self.permute_(perm))
        finalize = AbstractTensor._pre_autograd(
            "permute", [self], params={"perm": list(perm)}
        )
        return finalize(result)

    try:
        backend_cls = BACKEND_REGISTRY.get("numpy")
        if backend_cls is not None:
            numpy_tensor = backend_cls(track_time=getattr(self, "track_time", False))
            numpy_tensor = numpy_tensor.ensure_tensor(self.data)
            permuted = numpy_tensor.data.transpose(perm)
            perm_tensor = backend_cls(track_time=getattr(self, "track_time", False))
            perm_tensor.data = permuted
            finalize = AbstractTensor._pre_autograd(
                "permute", [self], params={"perm": list(perm)}
            )
            return finalize(perm_tensor.to_backend(self))
    except Exception:
        pass

    raise NotImplementedError("permute fallback not implemented for this backend.")


def unsqueeze(self, dim: int) -> "AbstractTensor":
    """Return a tensor with an inserted dimension of size 1 at ``dim``."""
    from ..abstraction import AbstractTensor

    if hasattr(self, "unsqueeze_"):
        result = _wrap_result(self, self.unsqueeze_(dim))
        finalize = AbstractTensor._pre_autograd(
            "reshape", [self], params={"new_shape": result.shape}
        )
        return finalize(result)

    raise NotImplementedError("Unsqueeze fallback not implemented for this backend.")

def swapaxes(self, axis1: int, axis2: int) -> "AbstractTensor":
    from ..abstraction import AbstractTensor

    shape = getattr(self, "shape", ())
    ndim = len(shape)
    if not (-ndim <= axis1 < ndim) or not (-ndim <= axis2 < ndim):
        raise ValueError("axis1 or axis2 out of range")
    axis1 %= ndim
    axis2 %= ndim

    if hasattr(self, "swapaxes_"):
        perm = list(range(ndim))
        perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
        finalize = AbstractTensor._pre_autograd(
            "permute", [self], params={"perm": perm}
        )
        return finalize(_wrap_result(self, self.swapaxes_(axis1, axis2)))
    raise NotImplementedError("swapaxes fallback not implemented for this backend.")

def squeeze(self, dim: int | None = None) -> "AbstractTensor":
    """Return a tensor with all (or one) dimensions of size 1 removed."""
    from ..abstraction import AbstractTensor

    if hasattr(self, "squeeze_"):
        result = _wrap_result(self, self.squeeze_(dim))
        finalize = AbstractTensor._pre_autograd(
            "reshape", [self], params={"new_shape": result.shape}
        )
        return finalize(result)
    raise NotImplementedError("Squeeze fallback not implemented for this backend.")


def flatten(self) -> "AbstractTensor":
    """Return a flattened version of the tensor as an AbstractTensor."""
    from ..abstraction import AbstractTensor, BACKEND_REGISTRY

    if hasattr(self, "flatten_"):
        finalize = AbstractTensor._pre_autograd(
            "reshape", [self], params={"new_shape": (-1,)}
        )
        return finalize(_wrap_result(self, self.flatten_()))
    # Fast common path: use reshape(-1). The new normalizer makes this robust.
    try:
        return self.reshape(-1)
    except Exception:
        pass
    finalize = AbstractTensor._pre_autograd(
        "reshape", [self], params={"new_shape": (-1,)}
    )
    if hasattr(self, "flatten_"):
        return finalize(_wrap_result(self, self.flatten_()))
    try:
        backend_cls = BACKEND_REGISTRY.get("numpy")
        if backend_cls is not None:
            numpy_tensor = backend_cls(track_time=getattr(self, "track_time", False))
            numpy_tensor = numpy_tensor.ensure_tensor(self.data)
            if hasattr(numpy_tensor.data, "flatten"):
                flat_data = numpy_tensor.data.flatten()
                flat_tensor = backend_cls(track_time=getattr(self, "track_time", False))
                flat_tensor.data = flat_data
                return finalize(flat_tensor.to_backend(self))
    except Exception:
        pass
    try:
        backend_cls = BACKEND_REGISTRY.get("pure_python")
        if backend_cls is not None:
            py_tensor = backend_cls(track_time=getattr(self, "track_time", False))
            py_tensor = py_tensor.ensure_tensor(self.data)
            def _flatten(data):
                if not isinstance(data, list):
                    return [data]
                return [item for sublist in data for item in _flatten(sublist)]
            flat_data = _flatten(py_tensor.data)
            flat_tensor = backend_cls(track_time=getattr(self, "track_time", False))
            flat_tensor.data = flat_data
            return finalize(flat_tensor.to_backend(self))
    except Exception:
        pass
    def _flatten(data):
        if not isinstance(data, list):
            return [data]
        return [item for sublist in data for item in _flatten(sublist)]
    flat_data = _flatten(self.tolist())
    out = type(self)(track_time=getattr(self, "track_time", False))
    out.data = flat_data
    return finalize(out)


def repeat(self, repeats: Any = None, dim: int = 0) -> "AbstractTensor":
    """Repeat ``self`` along ``dim`` ``repeats`` times."""
    return _wrap_result(self, self.repeat_(repeats, dim))

def repeat_interleave(self, repeats: int = 1, dim: Optional[int] = None) -> "AbstractTensor":
    return _wrap_result(self, self.repeat_interleave_(repeats, dim))

