"""Resident AbstractTensor adapter for the GLSL compute backend.

The backend implements the canonical primitive vocabulary through one
``_apply_operator__`` dispatch. Higher mathematics remains in AbstractTensor
and composes those primitives. Host arrays are used only to stage creation or
to service an explicit ``numpy``/``tolist``/``item`` boundary; arithmetic
results remain :class:`GLChunk` objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..abstraction import AbstractTensor, register_backend
from ..abstraction_methods.indexing import flat_index_ids
from .glsl_backend import (
    GLChunk,
    _normalize_dtype,
    arange_chunk,
    cat_chunks,
    cumsum_chunk,
    expand_chunk,
    full_chunk,
    gather_offsets_chunk,
    index_assign_index_chunk,
    index_assign_offsets_chunk,
    index_select_chunk,
    matmul_chunks,
    permute_chunk,
    repeat_chunk,
    reduce_chunk,
    reshape_chunk,
    slice_axis_chunk,
    run_op,
    stack_chunks,
    topk_chunks,
)


def _raw(value: Any) -> Any:
    return value.data if isinstance(value, AbstractTensor) else value


class GLSLTensorOperations(AbstractTensor):
    """AbstractTensor operations whose native storage is a resident GL SSBO."""

    supports_native_batched_matmul = True
    tensor_type_ = GLChunk
    float_dtype_ = np.dtype(np.float32)
    long_dtype_ = np.dtype(np.int32)
    bool_dtype_ = np.dtype(np.bool_)

    def _apply_operator__(self, op: str, left: Any, right: Any):
        if op in {"matmul", "rmatmul", "imatmul"}:
            return matmul_chunks(left, right, reverse=op == "rmatmul")
        return run_op(op, left, right)

    # Creation and explicit transfer boundaries.
    def tensor_from_list_(self, data, dtype=None, device=None):
        return GLChunk.from_numpy(data, dtype=dtype)

    def empty_(self, size, dtype=None, device=None):
        return GLChunk(tuple(size), dtype=_normalize_dtype(dtype))

    def full_(self, size, fill_value, dtype=None, device=None):
        return full_chunk(size, fill_value, dtype=dtype)

    def zeros_(self, size, dtype=None, device=None):
        return full_chunk(size, 0, dtype=dtype)

    def ones_(self, size, dtype=None, device=None):
        return self.full_(size, 1, dtype=dtype, device=device)

    def zeros_like_(self, dtype=None, device=None):
        return self.zeros_(
            self.data.shape,
            dtype=self.data.dtype if dtype is None else dtype,
            device=device,
        )

    def ones_like_(self, dtype=None, device=None):
        return self.ones_(
            self.data.shape,
            dtype=self.data.dtype if dtype is None else dtype,
            device=device,
        )

    def full_like_(self, fill_value, dtype=None, device=None):
        return self.full_(
            self.data.shape,
            fill_value,
            dtype=self.data.dtype if dtype is None else dtype,
            device=device,
        )

    # ``arange`` is registered with the public creation helpers, but unlike the
    # shared random generators it directly requires a backend ``arange_`` hook.
    # The mature NumPy, Torch, JAX, Pure, and C backends each supply one.  This
    # is another member of the expanded backend surface being inventoried here,
    # not an elementwise primitive that failed to graft onto GLSL.
    def arange_(self, start, end, step=1, *, dtype=None, device=None):
        if device not in (None, "glsl", "gpu"):
            raise ValueError(f"GLSL backend cannot create data on {device!r}")
        return arange_chunk(start, end, step, dtype=dtype)

    def clone_(self, tensor=None):
        value = _raw(self.data if tensor is None else tensor)
        if not isinstance(value, GLChunk):
            value = GLChunk.from_numpy(value)
        if value.dtype.kind == "b":
            return run_op("logical_or", value, False)
        return run_op("add", value, 0)

    def to_device_(self, device):
        if device not in (None, "glsl", "gpu"):
            raise ValueError(f"GLSL backend cannot move data to {device!r}")
        return self.data.to_gpu()

    def get_device_(self):
        return "glsl"

    def get_dtype_(self):
        return self.data.dtype

    def sum_(self, dim=None, keepdim=False):
        return reduce_chunk(self.data, "sum", dim, keepdim)

    def mean_(self, dim=None, keepdim=False):
        return reduce_chunk(self.data, "mean", dim, keepdim)

    def cumsum_(self, dim=0):
        return cumsum_chunk(self.data, dim)

    def min_(self, dim=None, keepdim=False):
        return reduce_chunk(self.data, "min", dim, keepdim)

    def max_(self, dim=None, keepdim=False):
        return reduce_chunk(self.data, "max", dim, keepdim)

    def topk_(self, k, dim=-1):
        return topk_chunks(self.data, k, dim)

    def any_(self, dim=None):
        return reduce_chunk(self.data, "any", dim, False)

    def all_(self, dim=None):
        return reduce_chunk(self.data, "all", dim, False)

    def to_dtype_(self, dtype: Any = "float", tensor=None):
        value = _raw(self.data if tensor is None else tensor)
        target = _normalize_dtype(dtype)
        if value.dtype == target:
            # A no-op cast has the same aliasing contract as Torch's native
            # ``Tensor.to(dtype)`` fast path. Preserve resident storage instead
            # of launching an identity arithmetic kernel.
            return value.view(value.shape)
        source_kind = value.dtype.kind
        target_kind = target.kind
        if target_kind == "b":
            return run_op("not_equal", value, 0)
        if target_kind == "f":
            return run_op(
                "uitofp" if source_kind in {"u", "b"} else "sitofp",
                value,
            )
        if target_kind == "u":
            return run_op("fptoui" if source_kind == "f" else "zext", value)
        return run_op("fptosi" if source_kind == "f" else "sext", value)

    # Public type conveniences delegate to the same primitive cast path. GLSL
    # storage is intentionally 32-bit, so ``double``/``long`` preserve the
    # backend contract rather than claiming unsupported 64-bit SSBO scalars.
    def long_cast_(self):
        return self.to_dtype_("int64")

    def float_(self):
        return self.to_dtype_("float32")

    def double_(self):
        return self.to_dtype_("float64")

    def int_(self):
        return self.to_dtype_("int32")

    def long_(self):
        return self.to_dtype_("int64")

    def bool_(self):
        return self.to_dtype_("bool")

    def numpy(self):
        return self.data.numpy()

    def __array__(self, dtype=None):
        array = self.data.numpy()
        return array if dtype is None else array.astype(dtype)

    def tolist_(self):
        return self.data.numpy().tolist()

    def item_(self):
        if self.data.count != 1:
            raise ValueError("only one-element GLSL tensors can be scalars")
        return self.data.numpy().reshape(-1)[0].item()

    def get_item_(self, data, index):
        """Lower basic and one-array indexing to resident axis selections."""
        items = list(index) if isinstance(index, tuple) else [index]
        ellipses = sum(item is Ellipsis for item in items)
        if ellipses > 1:
            raise IndexError("an index can only have a single ellipsis")
        consumed = sum(item is not None and item is not Ellipsis for item in items)
        if consumed > data.ndim:
            raise IndexError("too many indices for tensor")
        if ellipses:
            position = items.index(Ellipsis)
            items[position:position + 1] = [
                slice(None)
            ] * (data.ndim - consumed)
        else:
            items.extend([slice(None)] * (data.ndim - consumed))

        current = data
        output_axis = 0
        advanced_count = 0
        for item in items:
            if item is None:
                shape = (
                    current.shape[:output_axis]
                    + (1,)
                    + current.shape[output_axis:]
                )
                current = current.view(shape)
                output_axis += 1
                continue
            axis_size = current.shape[output_axis]
            if isinstance(item, (int, np.integer)):
                value = int(item)
                if value < -axis_size or value >= axis_size:
                    raise IndexError("tensor index out of range")
                value %= axis_size
                selected = slice_axis_chunk(
                    current, output_axis, value, 1, 1
                )
                current = selected.view(
                    selected.shape[:output_axis]
                    + selected.shape[output_axis + 1:]
                )
                continue
            if isinstance(item, slice):
                positions = range(*item.indices(axis_size))
                current = slice_axis_chunk(
                    current,
                    output_axis,
                    positions.start,
                    positions.step,
                    len(positions),
                )
                output_axis += 1
                continue

            advanced_count += 1
            if advanced_count > 1:
                raise NotImplementedError(
                    "at most one integer-array index is currently supported"
                )
            raw = _raw(item)
            if isinstance(raw, GLChunk):
                index_chunk = raw
            else:
                array = np.asarray(
                    raw.tolist() if hasattr(raw, "tolist") else raw
                )
                if array.dtype.kind not in "iu":
                    raise TypeError("advanced tensor indices must be integers")
                normalized = np.where(array < 0, array + axis_size, array)
                if np.any((normalized < 0) | (normalized >= axis_size)):
                    raise IndexError("tensor index out of range")
                index_chunk = GLChunk.from_numpy(
                    normalized, dtype=np.int32
                )
            current = index_select_chunk(current, output_axis, index_chunk)
            output_axis += index_chunk.ndim
        return current

    def set_item_(self, data, index, value):
        """Normalize indexing once, then assign source values on the GPU."""

        raw_index = _raw(index)
        if isinstance(raw_index, tuple):
            raw_index = tuple(_raw(item) for item in raw_index)
        raw_value = _raw(value)
        if not isinstance(raw_value, GLChunk):
            raw_value = GLChunk.from_numpy(
                np.asarray(raw_value, dtype=data.dtype), dtype=data.dtype
            )
        elif raw_value.dtype != data.dtype:
            temporary = type(self)()
            temporary.data = raw_value
            raw_value = temporary.to_dtype_(data.dtype)
        flat_index = (
            raw_index[0]
            if isinstance(raw_index, tuple) and len(raw_index) == 1
            else raw_index
        )
        if data.ndim == 1 and isinstance(flat_index, GLChunk):
            index_assign_index_chunk(data, flat_index, raw_value)
        else:
            def materialize_index(item):
                if isinstance(item, GLChunk):
                    return item.numpy()
                if isinstance(item, tuple):
                    return tuple(materialize_index(part) for part in item)
                if isinstance(item, list):
                    return [materialize_index(part) for part in item]
                return item

            offsets = flat_index_ids(
                materialize_index(raw_index), data.shape
            )
            index_assign_offsets_chunk(data, offsets, raw_value)

    # Backend-required structural surface.
    #
    # This work is indirectly discovering that a usable backend contract is
    # wider than ``_apply_operator__`` and its primitive mathematical
    # vocabulary.  AbstractTensor owns the public composition and autograd
    # semantics, but storage metadata, accessors, and layout transformations
    # still require hooks from the backend that owns the representation.
    #
    # In particular, AbstractTensor's ``reshape_`` is an explicit raising hook,
    # not a universal implementation.  NumPy, Torch, JAX, Pure, and C each
    # supply it; ``view`` and the common ``flatten`` route use it as well.
    # ``cat`` and ``stack`` likewise have always dispatched via backend
    # ``cat_``/``stack_`` hooks rather than being missing reshape/indexing
    # grafts.  NumPy and Torch expose them through different native entry
    # points, reinforcing the separation.
    #
    # For GLSL this separation is useful rather than merely historical.
    # Reshape is a zero-copy shared-storage view; permute, cat, and stack lower
    # their complete arbitrary-rank layout transforms to one purpose-built
    # compute shader and one device-planned dispatch. ``stack`` could be
    # expressed as unsqueeze-plus-cat, but that would allocate an avoidable
    # intermediate and hide future GPU-specific opportunities such as tiled or
    # subgroup copies.
    def reshape_(self, shape):
        return reshape_chunk(self.data, shape)

    def expand_(self, shape):
        return expand_chunk(self.data, shape)

    def repeat_(self, repeats=None, dim=0):
        return repeat_chunk(self.data, repeats, dim)

    def permute_(self, dims):
        return permute_chunk(self.data, dims)

    def transpose_(self, dim0, dim1):
        rank = self.data.ndim
        if not -rank <= dim0 < rank or not -rank <= dim1 < rank:
            raise ValueError("dim0 or dim1 out of range")
        dim0 %= rank
        dim1 %= rank
        dims = list(range(rank))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return permute_chunk(self.data, dims)

    def swapaxes_(self, axis1, axis2):
        return self.transpose_(axis1, axis2)

    def stack_(self, tensors, dim=0):
        return stack_chunks([_raw(tensor) for tensor in tensors], dim=dim)

    def cat_(self, tensors, dim=0):
        return cat_chunks([_raw(tensor) for tensor in tensors], dim=dim)

    # Storage metadata is not an operator implementation.
    def get_shape(self):
        return self.data.shape

    def get_ndims(self):
        return self.data.ndim

    def numel_(self):
        return self.data.count

    def nbytes_(self):
        return self.data.nbytes

    # AbstractTensor's older convenience methods call these hooks. They all
    # route back through the same primitive dispatcher.
    def neg_(self):
        return run_op("neg", self.data)

    def abs_(self):
        return run_op("abs", self.data)

    def invert_(self):
        return run_op("invert", self.data)

    def round_(self, n=None):
        if n not in (None, 0):
            scale = float(10 ** int(n))
            return run_op(
                "truediv",
                run_op("round", run_op("mul", self.data, scale)),
                scale,
            )
        return run_op("round", self.data)

    def trunc_(self):
        return run_op("trunc", self.data)

    def floor_(self):
        return run_op("floor", self.data)

    def ceil_(self):
        return run_op("ceil", self.data)

    def sqrt_(self):
        return run_op("sqrt", self.data)

    def exp_(self):
        return run_op("exp", self.data)

    def log_(self):
        return run_op("log", self.data)

    def logical_not_(self):
        return run_op("logical_not", self.data)

    def isfinite_(self):
        return run_op("isfinite", self.data)

    def isnan_(self):
        return run_op("isnan", self.data)

    def isinf_(self):
        return run_op("isinf", self.data)

    def greater_(self, value):
        return run_op("greater", self.data, _raw(value))

    def greater_equal_(self, value):
        return run_op("greater_equal", self.data, _raw(value))

    def less_(self, value):
        return run_op("less", self.data, _raw(value))

    def less_equal_(self, value):
        return run_op("less_equal", self.data, _raw(value))

    def equal_(self, value):
        return run_op("equal", self.data, _raw(value))

    def not_equal_(self, value):
        return run_op("not_equal", self.data, _raw(value))

    def maximum_(self, other):
        return run_op("maximum", self.data, _raw(other))

    def minimum_(self, other):
        return run_op("minimum", self.data, _raw(other))

    # Override the legacy valuewise shims: a backend with native vector
    # primitives must not flatten and read back to Python.
    def sign(self):
        return self._apply_operator("sign", self, None)

    def __invert__(self):
        # AbstractTensor's arithmetic promotion treats integer ``~`` as a
        # bitwise operation. GLSL bool storage is uint-backed, but its logical
        # dtype must retain NumPy/Torch boolean inversion semantics.
        op = "logical_not" if self.data.dtype.kind == "b" else "invert"
        return self._apply_operator(op, self, None)

    def __eq__(self, other):
        return self._apply_operator("equal", self, other)

    def __ne__(self, other):
        return self._apply_operator("not_equal", self, other)

    def __lt__(self, other):
        return self._apply_operator("less", self, other)

    def __le__(self, other):
        return self._apply_operator("less_equal", self, other)

    def __gt__(self, other):
        return self._apply_operator("greater", self, other)

    def __ge__(self, other):
        return self._apply_operator("greater_equal", self, other)

    def __and__(self, other):
        return self._apply_operator("bitand", self, other)

    def __or__(self, other):
        return self._apply_operator("bitor", self, other)

    def __xor__(self, other):
        return self._apply_operator("bitxor", self, other)

    def logical_and(self, other):
        return self._apply_operator("logical_and", self, other)

    def logical_or(self, other):
        return self._apply_operator("logical_or", self, other)


register_backend("glsl", GLSLTensorOperations)
